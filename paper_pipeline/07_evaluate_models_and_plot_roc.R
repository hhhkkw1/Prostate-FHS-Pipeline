# ROC, FHS association and probability delta/IDI. Step 6 supplies shared metrics.
options(stringsAsFactors=FALSE)
# R >= 4.2 on Windows supports UTF-8. This also handles Unicode workbook
# paths when the launching shell has supplied an unsupported C.UTF-8 locale.
if (.Platform$OS.type=='windows' && !isTRUE(l10n_info()[['UTF-8']])) {
  invisible(Sys.setlocale('LC_CTYPE','English_United States.utf8'))
}
required <- c('jsonlite','readxl','pROC','openxlsx','ggplot2')
missing <- required[!vapply(required,requireNamespace,logical(1),quietly=TRUE)]
if(length(missing)) stop('Install R dependencies first: ',paste(missing,collapse=', '))
args <- commandArgs(trailingOnly=TRUE)
script_arg <- grep('^--file=',commandArgs(),value=TRUE)
script_dir <- if(length(script_arg)) dirname(normalizePath(sub('^--file=','',script_arg[1]))) else getwd()
config_path <- if(length(args) && args[1]!='--no-plots') args[1] else file.path(script_dir,'config','pipeline_config.json')
cfg <- jsonlite::fromJSON(config_path)
root <- dirname(dirname(normalizePath(config_path)))
resolve <- function(p) if(grepl('^([A-Za-z]:|/|\\\\)',p)) p else file.path(root,p)
file_path <- resolve(cfg$paths$prediction_output_excel)
output <- resolve(cfg$paths$evaluation_output_excel)
plot_dir <- resolve(cfg$paths$plot_dir)
dir.create(plot_dir,recursive=TRUE,showWarnings=FALSE)
dir.create(dirname(output),recursive=TRUE,showWarnings=FALSE)
make_plots <- !('--no-plots' %in% args)
dpi <- if(is.null(cfg$analysis$dpi)) 600 else cfg$analysis$dpi
split <- cfg$columns$split_col; outcome <- cfg$columns$outcome_col
datasets <- unname(unlist(cfg$dataset_values[c('training','internal_val','external_val')]))
codes <- c('Training','Internal_Val','External_Val')
display_names <- c('Training','Internal validation','Cross-site validation')
cohort_labeller <- ggplot2::labeller(Cohort=setNames(display_names,codes))
df <- as.data.frame(readxl::read_excel(file_path))
df[[split]] <- datasets[match(tolower(trimws(df[[split]])),tolower(datasets))]
excluded_n <- sum(is.na(df[[split]]))
df <- df[!is.na(df[[split]]),,drop=FALSE]
if(!('pred_prob_ImageModel' %in% names(df)) && 'pred_prob_Image' %in% names(df)) df$pred_prob_ImageModel <- df$pred_prob_Image
cols <- c('pred_prob_Habitat','pred_prob_CAPRA','pred_prob_Clinical','pred_prob_Clin_PIRADS','pred_prob_Combined','pred_prob_ImageModel')
labels <- c('FHS Biomarker','CAPRA Benchmark','Clinical Model','Clinical-Imaging Model','FHS-Integrated Model','Image Model')
if(length(setdiff(c(split,outcome,cols),names(df)))) stop('Missing required analysis columns.')
for(c in c(outcome,cols)) df[[c]] <- as.numeric(df[[c]])
if(anyNA(df[[outcome]]) || !all(df[[outcome]] %in% 0:1)) stop('Invalid included AP outcomes.')
if(!all(vapply(df[cols],function(p) all(is.finite(p)&p>=0&p<=1),logical(1)))) stop('Invalid probabilities; paired analyses require common patients.')
metrics_path <- file.path(plot_dir,'Binary_Model_Metrics.csv')
provenance_path <- file.path(plot_dir,'Binary_Report_Provenance.json')
if(!file.exists(metrics_path) || !file.exists(provenance_path)) stop('Run step 6 first (or --analysis-only).')
provenance <- jsonlite::fromJSON(provenance_path)
if(unname(tools::md5sum(file_path))!=provenance$prediction_md5) stop('Prediction workbook changed; regenerate step 6 reports.')
expected_seed <- if(is.null(cfg$analysis$seed)) 123 else cfg$analysis$seed
expected_bootstrap <- if(is.null(cfg$analysis$brier_bootstrap)) 2000 else cfg$analysis$brier_bootstrap
if(provenance$seed!=expected_seed || provenance$brier_bootstrap!=expected_bootstrap) stop('Brier bootstrap configuration changed; regenerate step 6 reports.')
metrics <- read.csv(metrics_path,check.names=FALSE)
wb <- openxlsx::createWorkbook()
add <- function(name,table) {openxlsx::addWorksheet(wb,name);openxlsx::writeData(wb,name,table)}
checks <- list()
for(i in seq_along(datasets)) {
  ds <- df[df[[split]]==datasets[i],,drop=FALSE]; y <- ds[[outcome]]
  if(min(sum(y==0),sum(y==1))<2) stop('Need two events/non-events in ',datasets[i])
  rows <- metrics[metrics$Dataset_Type==codes[i],,drop=FALSE]
  rocs <- lapply(cols,function(c) pROC::roc(y,ds[[c]],levels=c(0,1),direction='<',quiet=TRUE));names(rocs)<-cols
  for(j in seq_along(cols)) {
    c <- cols[j]; row <- rows[rows$Probability_Column==c,,drop=FALSE]
    if(nrow(row)!=1 || row$N!=nrow(ds)) stop('Metric denominator mismatch.')
    cm <- table(factor(y,levels=0:1),factor(as.integer(ds[[c]]>=row$Threshold_Used),levels=0:1))
    if(abs(as.numeric(pROC::auc(rocs[[c]]))-row$AUC)>1e-10 || !all(as.numeric(t(cm))==as.numeric(unlist(row[c('TN','FP','FN','TP')])))) stop('R/Python AUC or confusion count mismatch.')
    if(j!=5) {
      rp <- as.numeric(pROC::roc.test(rocs[[c]],rocs[['pred_prob_Combined']],method='delong',paired=TRUE)$p.value)
      if(is.finite(rp) && abs(rp-row[['DeLong p']])>1e-8) stop('R/Python DeLong mismatch.')
    }
    checks[[length(checks)+1]] <- data.frame(Cohort=datasets[i],Model=labels[j],N=nrow(ds),AUC_matches=TRUE,CM_matches=TRUE)
  }
  add(paste0(codes[i],'_Metrics'),rows)
  add(paste0(codes[i],'_Comparison'),rows[rows$Model!=labels[5],c('Model','DeLong p','Brier_Difference_vs_Ref','Brier_Diff_CI_Lower','Brier_Diff_CI_Upper','Brier_p'),drop=FALSE])
  if(make_plots) {
    tiff(file.path(plot_dir,paste0('ROC_5models_',codes[i],'.tiff')),width=4,height=4,units='in',res=dpi,compression='lzw')
    colors <- c('#FF7F0E','#1F77B4','#2CA02C','#9467BD','black')
    plot(rocs[[cols[5]]],col='black',lwd=2,legacy.axes=TRUE,xlab='1 - Specificity',ylab='Sensitivity',main=display_names[i])
    for(j in 1:4) plot(rocs[[cols[j]]],col=colors[j],lwd=1.5,add=TRUE)
    legend('bottomright',legend=paste0(labels[1:5],' (',sprintf('%.3f',vapply(rocs[1:5],function(r) as.numeric(pROC::auc(r)),numeric(1))),')'),col=colors,lwd=1.5,cex=.65,bty='n')
    dev.off()
  }
}
add('Cross_Language_Checks',do.call(rbind,checks))
# Association is training-only by default. Pooling must be explicitly configured
# as exploratory; biological holdout cannot be included here.
association_cohorts <- unlist(cfg$analysis$association_cohorts)
if(!length(association_cohorts)) association_cohorts <- cfg$dataset_values$training
if(!all(association_cohorts %in% datasets)) stop('Invalid association cohorts.')
keys <- c('habitat_score','age','psa','t_stage','gg_bp','pct_pos','pirads')
vars <- c('FHS','Age','PSA','T_stage','GG','pct_pos','PIRADS')
names_actual <- unname(unlist(cfg$clinical_columns[keys]))
if(length(setdiff(names_actual,names(df)))) stop('Missing association covariates.')
ad <- df[df[[split]] %in% association_cohorts,c(outcome,names_actual),drop=FALSE]
names(ad) <- c('Outcome',vars);ad[]<-lapply(ad,as.numeric)
complete <- complete.cases(ad)&apply(as.matrix(ad),1,function(x) all(is.finite(x)))
excluded_association <- sum(!complete);ad<-ad[complete,,drop=FALSE]
if(length(unique(ad$Outcome))!=2) stop('Association needs both outcome classes.')
extract <- function(fit,kind) {
  if(!fit$converged) stop('Association fit did not converge.')
  co <- summary(fit)$coefficients[-1,,drop=FALSE]
  if(nrow(co)!=length(coef(fit))-1 || any(!is.finite(co))) stop('Undefined/aliased association coefficients.')
  data.frame(Variable=rownames(co),Analysis=kind,Beta=co[,1],SE=co[,2],OR=exp(co[,1]),CI_lower=exp(co[,1]-qnorm(.975)*co[,2]),
             CI_upper=exp(co[,1]+qnorm(.975)*co[,2]),P=co[,4],N=nrow(ad),Events=sum(ad$Outcome),row.names=NULL)
}
uni <- do.call(rbind,lapply(vars,function(v) extract(glm(reformulate(v,response='Outcome'),data=ad,family=binomial()),'Univariable')))
multi <- extract(glm(reformulate(vars,response='Outcome'),data=ad,family=binomial()),'Multivariable')
add('FHS_Univariable',uni);add('FHS_Multivariable',multi)
fmt_p <- function(p) ifelse(p<.001,'<0.001',sprintf('%.3f',p))
formatted <- data.frame(Variable=vars,OR_Uni=sprintf('%.3f',uni$OR),CI_Uni=sprintf('%.3f–%.3f',uni$CI_lower,uni$CI_upper),P_Uni=fmt_p(uni$P),
                        OR_Multi=sprintf('%.3f',multi$OR),CI_Multi=sprintf('%.3f–%.3f',multi$CI_lower,multi$CI_upper),P_Multi=fmt_p(multi$P))
add('FHS_Formatted',formatted)
# Binary continuous IDI is descriptive: E(delta|AP) - E(delta|non-AP).
delta_list<-list();idi_list<-list()
for(i in seq_along(datasets)) {
  ds<-df[df[[split]]==datasets[i],,drop=FALSE];y<-ds[[outcome]]
  for(j in 1:4) {
    delta<-ds$pred_prob_Combined-ds[[cols[j]]]
    delta_list[[length(delta_list)+1]]<-data.frame(Cohort=codes[i],Comparator=labels[j],Outcome=ifelse(y==1,'AP','Non-AP'),Delta=delta)
    a<-mean(delta[y==1]);b<-mean(delta[y==0])
    idi_list[[length(idi_list)+1]]<-data.frame(Cohort=codes[i],Comparator=labels[j],N=length(y),Events=sum(y),Mean_delta_AP=a,Mean_delta_NonAP=b,IDI=a-b)
  }
}
delta<-do.call(rbind,delta_list);idi<-do.call(rbind,idi_list)
add('IDI_Descriptive',idi);write.csv(idi,file.path(plot_dir,'IDI_Descriptive.csv'),row.names=FALSE)
delta$Cohort <- factor(delta$Cohort,levels=codes)
idi$Cohort <- factor(idi$Cohort,levels=codes)
if(make_plots) {
  for(tag in c('AllCohorts','ValidationOnly')) {
    dd<-if(tag=='AllCohorts') delta else delta[delta$Cohort!='Training',,drop=FALSE]
    ii<-if(tag=='AllCohorts') idi else idi[idi$Cohort!='Training',,drop=FALSE]
    p<-ggplot2::ggplot(dd,ggplot2::aes(x=Delta,color=Outcome,fill=Outcome))+ggplot2::geom_density(alpha=.15)+
      ggplot2::geom_vline(xintercept=0,linetype='dashed')+ggplot2::facet_grid(Cohort~Comparator,labeller=cohort_labeller)+ggplot2::theme_bw()+
      ggplot2::labs(x='Probability difference: integrated minus comparator',y='Density')
    ggplot2::ggsave(file.path(plot_dir,paste0('Probability_Delta_',tag,'.tiff')),p,width=14,height=7,dpi=dpi,compression='lzw')
    p2<-ggplot2::ggplot(ii,ggplot2::aes(x=Comparator,y=IDI,fill=Comparator))+ggplot2::geom_col()+ggplot2::facet_wrap(~Cohort,labeller=cohort_labeller)+
      ggplot2::theme_bw()+ggplot2::theme(axis.text.x=ggplot2::element_text(angle=30,hjust=1),legend.position='none')+
      ggplot2::labs(x=NULL,y='IDI (descriptive point estimate)')
    ggplot2::ggsave(file.path(plot_dir,paste0('IDI_',tag,'.tiff')),p2,width=12,height=4,dpi=dpi,compression='lzw')
  }
}
methods<-data.frame(Item=c('Binary metrics','ROC','Brier comparison','Association cohorts','Association N','Association exclusions','Association method','Units','Excluded holdout','IDI'),
                    Details=c('Shared step 6 results; independently verified by R','AP=1; direction=<; never auto-reverse AUC',
                              paste0('Normal test using paired bootstrap SE; B=',provenance$brier_bootstrap,'; seed=',provenance$seed,'; fixed predictions; unadjusted'),
                              paste(association_cohorts,collapse=', '),nrow(ad),excluded_association,'Unpenalized GLM; Wald CI and two-sided Wald P; common complete cases',
                              'T/GG/PI-RADS per ordinal step; pct_pos per fraction 0–1; FHS per raw unit',excluded_n,'Descriptive point estimate; no significance claim'))
add('Methods',methods)
openxlsx::saveWorkbook(wb,output,overwrite=TRUE)
cat('Saved:',output,'\nAll 18 R/Python binary checks passed.\n')
