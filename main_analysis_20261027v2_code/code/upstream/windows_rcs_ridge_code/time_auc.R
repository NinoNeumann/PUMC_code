# Cumulative/dynamic IPCW ROC for right-censored all-cause survival.
# Marginal KM censoring; event cases by t, event-free controls after t.
# Paired delete-one jackknife includes re-estimation of censoring KM.
# This is inference on saved held-out predictions, NOT bootstrap validation.
auc29_cache <- new.env(parent=emptyenv())
auc29_weights <- function(d, horizon) {
  key<-paste(horizon,paste(d$patient_id,d$Survival_Days,d$Death,sep=':',collapse='|'),sep=':')
  if(exists(key,auc29_cache,inherits=FALSE))return(get(key,auc29_cache))
  T<-d$Survival_Days;E<-d$Death;n<-length(T)
  cases<-which(T<=horizon&E==1);controls<-which(T>horizon)
  if(length(cases)<2||length(controls)<2)stop('AUC requires >=2 observed cases and >=2 controls for delete-one inference at ',horizon,' days')
  # Column 1 full sample, columns 2..n+1 delete patient 1..n.
  W<-matrix(0,length(cases),n+1)
  for(del in 0:n){
    ix<-if(del==0)seq_len(n) else setdiff(seq_len(n),del)
    ct<-sort(unique(T[ix][E[ix]==0&T[ix]<=horizon]))
    fac<-vapply(ct,function(u){Y<-sum(T[ix]>=u);dc<-sum(T[ix]==u&E[ix]==0);1-dc/Y},numeric(1))
    G<-cumprod(fac)
    left<-vapply(T[cases],function(u){v<-which(ct<u);if(length(v))G[max(v)] else 1},numeric(1))
    if(any(left<=0)|| (length(G)&&tail(G,1)<=0))stop('Censoring KM lacks support at requested horizon')
    W[,del+1]<-1/left
    if(del%in%cases)W[match(del,cases),del+1]<-0
  }
  z<-list(cases=cases,controls=controls,W=W,n=n)
  assign(key,z,auc29_cache);z
}
auc29_one <- function(d,score,horizon,grid=seq(0,1,length.out=201)){
  stopifnot(length(score)==nrow(d),all(is.finite(score)))
  w<-auc29_weights(d,horizon);ca<-w$cases;co<-w$controls;n<-w$n
  H<-outer(score[ca],score[co],function(x,y)(x>y)+.5*(x==y))
  ranks<-matrix(rowSums(H),length(ca),n+1)
  for(j in seq_along(co))ranks[,co[j]+1]<-ranks[,co[j]+1]-H[,j]
  nc<-rep(length(co),n+1);nc[co+1]<-nc[co+1]-1
  est<-colSums(w$W*ranks)/(colSums(w$W)*nc)
  influence<-(mean(est[-1])-est[-1])*sqrt((n-1)/n)
  # Exact ROC at tied score blocks, with zero weight for early censoring.
  wtcase<-numeric(n);wtcase[ca]<-w$W[,1]/sum(w$W[,1])
  wtcontrol<-numeric(n);wtcontrol[co]<-1/length(co)
  o<-order(score,decreasing=TRUE);end<-c(which(diff(score[o])!=0),n)
  fp<-c(0,cumsum(wtcontrol[o])[end]);tp<-c(0,cumsum(wtcase[o])[end])
  # Linear interpolation of the polyline, retaining vertical ROC segments.
  # Grid is for plotting only; exact rank-based AUC above is reported.
  roc<-vapply(grid,function(x){
    eq<-which(abs(fp-x)<1e-12);if(length(eq))return(max(tp[eq]))
    lo<-max(which(fp<x));hi<-min(which(fp>x))
    tp[lo]+(tp[hi]-tp[lo])*(x-fp[lo])/(fp[hi]-fp[lo])
  },numeric(1))
  list(estimate=est[1],influence=influence,roc=roc,
       cases=length(ca),controls=length(co),n=n,
       exact_ROC=data.frame(FPR=fp,TPR=tp))
}

# Aggregate stratum AUCs using prespecified assessment-size weights; only
# within-fold rankings. Cluster repeated influence contributions by patient.
auc29_pair <- function(d,horizon){
  parts<-split(d,interaction(d$repeat_id,d$fold,drop=TRUE));den<-nrow(d)
  ids<-sort(unique(d$patient_id));ia<-ib<-setNames(numeric(length(ids)),ids)
  qa<-qb<-0;ra<-rb<-numeric(201);counts<-list()
  for(s in names(parts)){
    z<-parts[[s]];a<-auc29_one(z,z$score_cox,horizon);b<-auc29_one(z,z$score_rcs_ridge,horizon);w<-nrow(z)/den
    qa<-qa+w*a$estimate;qb<-qb+w*b$estimate
    ia[as.character(z$patient_id)]<-ia[as.character(z$patient_id)]+w*a$influence
    ib[as.character(z$patient_id)]<-ib[as.character(z$patient_id)]+w*b$influence
    ra<-ra+w*a$roc;rb<-rb+w*b$roc
    counts[[s]]<-data.frame(stratum=s,N=nrow(z),cases=a$cases,controls=a$controls)
  }
  list(metrics=data.frame(C_cox=qa,C_rcs_ridge=qb,difference_rcs_ridge_minus_cox=qb-qa,
    variance_cox=sum(ia^2),variance_rcs_ridge=sum(ib^2),covariance_cox_rcs_ridge=sum(ia*ib),variance_difference=sum((ib-ia)^2)),
    roc=data.frame(FPR=seq(0,1,length.out=201),ordinary_cox=ra,rcs_ridge_cox=rb),counts=do.call(rbind,counts))
}

evaluate29_predictions <- function(d,out,label,comparison=FALSE){
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  stopifnot(setequal(unique(d$imputation),1:20),!anyDuplicated(d[c('repeat_id','fold','imputation','patient_id')]))
  cm<-do.call(rbind,lapply(1:20,function(j)data.frame(imputation=j,cox29_paired_C(d[d$imputation==j,]))))
  cp<-cox29_pool_metrics(cm);write_csv(cm,file.path(out,'C_index_each_imputation.csv'));write_csv(cp,file.path(out,'C_index_pooled_and_difference.csv'))
  am<-curves<-counts<-list()
  for(t in c(365,730,1095))for(j in 1:20){
    z<-auc29_pair(d[d$imputation==j,],t)
    am[[length(am)+1]]<-data.frame(days=t,imputation=j,z$metrics)
    curves[[length(curves)+1]]<-data.frame(days=t,imputation=j,z$roc)
    if(j==1)counts[[length(counts)+1]]<-data.frame(days=t,z$counts)
  }
  am<-do.call(rbind,am);curves<-do.call(rbind,curves)
  ap<-do.call(rbind,lapply(c(365,730,1095),function(t){z<-cox29_pool_metrics(am[am$days==t,]);z$metric<-sub('C_','AUC_',z$metric,fixed=TRUE);z$interval_scope<-'Conditional paired patient-clustered delete-one jackknife (KM re-estimated) + Rubin; excludes model development uncertainty';data.frame(days=t,z)}))
  rc<-aggregate(cbind(ordinary_cox,rcs_ridge_cox)~days+FPR,data=curves,FUN=mean)
  write_csv(am,file.path(out,'AUC_each_imputation.csv'));write_csv(ap,file.path(out,'AUC_pooled_and_difference.csv'));write_csv(rc,file.path(out,'ROC_mean_coordinates.csv'));write_csv(do.call(rbind,counts),file.path(out,'AUC_case_control_counts.csv'))
  draw<-function(){par(mfrow=c(1,3),mar=c(4,4,3,1),oma=c(0,0,3,0));for(t in c(365,730,1095)){
    q<-rc[rc$days==t,];a<-ap[ap$days==t,];plot(q$FPR,q$ordinary_cox,type='l',lwd=2,col='#176A73',xlim=c(0,1),ylim=c(0,1),xlab='1 - specificity',ylab='Sensitivity',main=paste(t/365,'year'));lines(q$FPR,q$rcs_ridge_cox,lwd=2,col='#B76B24');abline(0,1,lty=2,col='gray60');legend('bottomright',legend=sprintf(if(comparison)c('Main AUC %.3f','Sensitivity AUC %.3f') else c('Cox AUC %.3f','RCS-Ridge-Cox AUC %.3f'),a$estimate[1:2]),col=c('#176A73','#B76B24'),lty=1,lwd=2,bty='n',cex=.85)};mtext(label,outer=TRUE,font=2)}
  png(file.path(out,'ROC_1_2_3_years.png'),width=2400,height=850,res=160);draw();dev.off()
  pdf(file.path(out,'ROC_1_2_3_years.pdf'),width=15,height=5.3);draw();dev.off()
  list(C=cp,AUC=ap)
}
