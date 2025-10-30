#GEOG71922:Assignment2
#Student_ID:11155827

#import packages
library(sp)         
library(rgeos)
library(terra)
library(sf)
library(zoom) 
library(raster)           
library(rgdal)           
library(rgeos)            
library(gdistance)       
library(igraph)           
library(fitdistrplus)    
library(mlr)
library(kernlab)
library(here)

# 1. Read data
# 1.1. Read sciurus data
data_dir <- here::here("data")
out_path  <- here::here("outputs")
dir.create(out_path, showWarnings = FALSE)
sciurus <- read.csv(here::here("data", "Sciurus.csv"))
head(sciurus)
class(sciurus)

#subset the data to only include points with complete coordinates.
table(is.na(sciurus$Latitude)) #True = 0, no missing value
#remove all points with uncertainty > 1000m
#remove all points with status=unconfirmed
statusCategories<-c('Accepted', 'Accepted - considered correct', 'Accepted - correct')
table(sciurus$Identification.verification.status %in% statusCategories) #True=6871, False=2072
table(sciurus$Coordinate.uncertainty_m<=1000) #True=6815, False=2128
sciurus<-subset(sciurus, sciurus$Identification.verification.status %in% statusCategories)
sciurus<-subset(sciurus, sciurus$Coordinate.uncertainty_m<=1000)
nrow(sciurus) #6602

# 1.2. Read greysquirrel data
greysquirrel <- read.csv(here::here("data", "Grey_squirrel_records.csv"))
head(greysquirrel)
class(greysquirrel)

table(is.na(greysquirrel$Latitude)) #True = 5
#subset the data to only include points with complete coordinates.
greysquirrel<-subset(greysquirrel,!is.na(greysquirrel$Latitude))

#remove all points with uncertainty > 1000m
#remove all points with status=unconfirmed
table(greysquirrel$Coordinate.uncertainty..m<=1000) #True=42898, False=38
table(greysquirrel$Identification.verification.status %in% statusCategories) #True=42931, False=5
greysquirrel<-subset(greysquirrel, greysquirrel$Coordinate.uncertainty..m<=1000)
greysquirrel<-subset(greysquirrel, greysquirrel$Identification.verification.status %in% statusCategories)
nrow(greysquirrel) #42893


#2. Create a vector layer from coordinates
#make spatial points layer
#create crs object
sciurus.latlong<-data.frame(x=sciurus$Longitude,y=sciurus$Latitude)
greysquirrel.latlong<-data.frame(x=greysquirrel$Longitude,y=greysquirrel$Latitude)
#Use coordinates object to create our spatial points object
sciurus.sp<-vect(sciurus.latlong,geom=c("x","y"))
greysquirrel.sp<-vect(greysquirrel.latlong,geom=c("x","y"))
#check that the points now have our desired crs. 
crs(sciurus.sp)<-"epsg:4326"
crs(greysquirrel.sp)<-"epsg:4326"


#3. Mapping land-cover
#read in our raster data
LCM <- rast(here::here("data", "LCMUK.tif"))
crs(LCM)

#Read in the .shp file, study area
studyArea <- rgdal::readOGR(here::here("data", "Sciurus_SA.shp"))

#Crop the raster
LCMCrop <- crop(LCM$LCMUK_1, studyArea)

#Plot the raster
par(mfrow=c(1,1))
plot(LCMCrop)

#extent for occurrence data
studyExtent<-c(-4.15,-2.65,56.68,57.415)
#crop the occurrence data
C1<-crop(sciurus.sp, studyExtent)
C2<-crop(greysquirrel.sp, studyExtent)

#project to change the coordinate reference system
sciurusFin<-terra::project(C1, crs("Sciurus_SA.prj"))
greysquirrelFin<-terra::project(C2, crs("Sciurus_SA.prj"))

#plot the occurence points
plot(sciurusFin, add=T, col='red')
plot(greysquirrelFin, add=T, col='blue')

set.seed(11) #random seed for reproduction

#extent for spatsample
extent<-c(270000.1,360000.9,755000.5,837000.5)
# spatSample data "back", the point areas where greysquirrel doesn’t occur
back.xy <- spatSample(LCMCrop, size=1000,as.points=TRUE, ext=extent) 

plot(LCMCrop)
plot(sciurusFin, add=T, col='red')
plot(greysquirrelFin, add=T, col='blue')
plot(back.xy, add=T)


#4. Characterizing point locations
Abs<-data.frame(crds(back.xy),Pres=0) #Absence(random)
Pres<-data.frame(crds(sciurusFin),Pres=1) #Presence(sciurus)
Pres.grey<-data.frame(crds(greysquirrelFin),Pres=2) #Presence(greysquirrel)
occData<-rbind(Pres,Abs,Pres.grey) #Occurrences data


#5. Re-classifying the raster
#access levels of the raster by treating them as categorical data ('factors' in R)
LCMCrop.factor<-as.factor(LCMCrop)

#reclass and RCmatrix to classify landcover
reclass <- c(0,1,rep(0,14))
RCmatrix<- cbind(levels(LCMCrop.factor)[[1]],reclass)
RCmatrix<-RCmatrix[,2:3] #we only need columns 2 and 3, drop IDs

eA<-extract(LCMCrop,back.xy)
eP<-extract(LCMCrop,sciurusFin)
eP2<-extract(LCMCrop,greysquirrelFin)

table(eA[,2]) # eA[,2]: landcover category, frequency: the no. of rows with each category number
table(eP[,2]) 
table(eP2[,2])

#plot histograms to find out the relationship between landcover types and occurences
par(mfrow=c(1,1))
hist(eA[,2],freq=FALSE,breaks=c(0:21),xlim=c(0,21),ylim=c(0,0.4))
dev.off()
hist(eP[,2],freq=FALSE,breaks=c(0:21),xlim=c(0,21),ylim=c(0,0.5), main="Red squirrel distribution", xlab="landcover type", ylab="proportion")
hist(eP2[,2],freq=FALSE,breaks=c(0:21),xlim=c(0,21),ylim=c(0,0.8), main="Grey squirrel distribution", xlab="landcover type", ylab="proportion")

#extract broadleaf from LCMCrop
broadleaf <- classify(LCMCrop, RCmatrix)
crs(broadleaf)<-crs(LCMCrop)
par(mfrow=c(1,1))
plot.new()
plot(broadleaf)
plot(sciurusFin,add=TRUE, col='red')
plot(greysquirrelFin, add=TRUE, col='blue')


#6. Buffer Analysis
#We set the sequence of radii to provide by using the seq() function:
radii<-seq(100,2000,by=100)

#res: store landcover, res2: store no.of greysquirrel nearby
res <- matrix(NA, nrow = nrow(occData), ncol = length(radii))
res2 <- matrix(NA, nrow = nrow(occData), ncol = length(radii))

#landBuffer to automatically calculate the broadleaf landcover 
#and number of greysquirrel nearby for each point in the study area.
landBuffer <- function(i){         
  occPoints<-vect(occData[i,],geom=c("x","y"),crs="epsg:27700")  
  #buffer each point
  occBuffer <- raster::buffer(occPoints, width=radii[j])
  #crop the landcover layer to the buffer extent
  bufferlandcover <- crop(broadleaf, occBuffer)
  #crop the greysquirrel occurrence points to the buffer extent
  greysquirrelcover<-crop(greysquirrelFin, occBuffer)
  # mask the above to the buffer
  masklandcover <- mask(bufferlandcover, occBuffer)            
  #sum landcover area
  landcoverArea <- sum(values(masklandcover),na.rm = TRUE)*625      
  #count number of greysquirrel nearby
  count<-nrow(greysquirrelcover)
  # convert to percentage cover
  percentcover <- landcoverArea/expanse(occBuffer)*100           
  return(c(percentcover,count))} #get the result

#get the results by for loop
for(i in 1:nrow(occData)){
  for(j in seq_along(radii)){
    res[i,j]<-landBuffer(i)[1] #store landcover
    res2[i,j]<-landBuffer(i)[2] #store no.of greysquirrel
  }
}

#save the result
#saveRDS(res, file="res")
#saveRDS(res2, file="res2")
#res<-readRDS("res")
#res2<-readRDS("res2")

#create data frame from the res matrix so we can do some statistics
occData$Pres<-as.integer(occData$Pres)

res3<-data.frame(cbind(res,res2,occData$Pres))
colnames(res3)<-c("w100","w200","w300","w400","w500","w600","w700","w800","w900","w1000",
                  "w1100","w1200","w1300","w1400","w1500","w1600","w1700","w1800","w1900","w2000",
                  "w100n","w200n","w300n","w400n","w500n","w600n","w700n","w800n","w900n","w1000n",
                  "w1100n","w1200n","w1300n","w1400n","w1500n","w1600n","w1700n","w1800n","w1900n","w2000n","Pres")

set.seed(11) #for reproduction
#simple 3x3 cv for buffers
perf_levelCV.buff = mlr::makeResampleDesc(method = "RepCV", folds = 3, reps = 3)

acc<-c() #to store accuracy values to compare

sampler <- function(i){ #function to set up task and learner
  j<-i+20
  task.buff = mlr::makeClassifTask(data=res3[,c(i,j,41)], target="Pres")
  lrnRF.buff = mlr::makeLearner("classif.randomForest",
                                predict.type = "response",
                                fix.factors.prediction = TRUE)
}

#get accuracy values for each buff size
sampler(1)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(2)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(3)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(4)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(5)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(6)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(7)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(8)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(9)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(10)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(11)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(12)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(13)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(14)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(15)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(16)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(17)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(18)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(19)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))
sampler(20)
cvRF = resample(learner = lrnRF.buff, task =task.buff,
                resampling = perf_levelCV.buff, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE)
acc<-c(acc,as.numeric(cvRF$aggr))

#dist labels for accuracy values
Dist<-c(100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000)

res4<-data.frame(cbind(Dist,acc)) #make a dataframe

#and plot
dev.off()
plot(res4$Dist, res4$acc, type = "b", frame = FALSE, pch = 19, 
     col = "red", xlab = "buffer", ylab = "accuracy")

#now, by finding the max accuracy 
#we can determine the optimum buffer size
opt<-res4[which(res4$acc==max(res4$acc)),]

#print the optimum buffer size with its accuracy value
opt #1200, 0.7594297


#7. Model training using cv and spcv
#dataset of 1200 buff size for training
dset<-data.frame(cbind(res3$w1200,res3$w1200n,res3$Pres,occData[,0:2]))
colnames(dset)<-c("landcover","adjgrey","Pres","x","y")
dset$Pres=as.integer(dset$Pres)
table(is.na(dset$area)) #TRUE=0

#task for training
task = mlr::makeClassifTask(data = dset, target = "Pres",coordinates=dset[,4:5])

#cv and spcv for training
perf_levelCV = mlr::makeResampleDesc(method = "RepCV", folds = 5, reps = 10)
perf_level_spCV = mlr::makeResampleDesc(method = "SpRepCV", folds = 5, reps = 10) 

#7.1. Random forest
lrnRF = mlr::makeLearner("classif.randomForest",
                               predict.type = "response",
                               fix.factors.prediction = TRUE)

#random forest cv
set.seed(11)
cvRF = resample(learner = lrnRF, task =task,
                resampling = perf_levelCV, 
                measures = mlr::acc, models=TRUE, keep.pred=TRUE) #acc:0.8086202

#random forest spcv
set.seed(11)
sp_cvRF = mlr::resample(learner = lrnRF, task =task,
                              resampling = perf_level_spCV, 
                              measures = mlr::acc, models=TRUE, keep.pred=TRUE) #0.5976022


#7.2 Support Vector Machine
lrn_ksvm = makeLearner("classif.ksvm",
                       predict.type = "response",
                       fix.factors.prediction = TRUE,
                       kernel = "rbfdot")

#svm cv
set.seed(11)
cvksvm = resample(learner = lrn_ksvm, task =task,
                  resampling = perf_levelCV, 
                  measures = mlr::acc) #0.7686356

#svm spcv
set.seed(11)
sp_cvksvm = resample(learner = lrn_ksvm, task =task,
                     resampling = perf_level_spCV, 
                     measures = mlr::acc) #0.6998442

#prediction by rf
rfMOD<-train(lrnRF,task)
prd.rfcv<-predict(rfMOD,task)
prd.t.rfcv<-cbind(data.frame(prd.rfcv)[,3],dset[,4:5])
colnames(prd.t.rfcv)<-c("Pred","x","y")

prd.abs.rfcv<-subset(prd.t.rfcv,prd.t.rfcv$Pred==0)
prd.sciurus.rfcv<-subset(prd.t.rfcv,prd.t.rfcv$Pred==1)
prd.grey.rfcv<-subset(prd.t.rfcv,prd.t.rfcv$Pred==2)

prd.abs.rfcv<-vect(prd.abs.rfcv, geom=c("x","y"))
prd.sciurus.rfcv<-vect(prd.sciurus.rfcv, geom=c("x","y"))
prd.grey.rfcv<-vect(prd.grey.rfcv, geom=c("x","y"))

plot(LCMCrop)

plot(prd.sciurus.rfcv, add=T, col='red')
plot(prd.grey.rfcv, add=T, col='blue')
plot(prd.abs.rfcv, add=T)

#prediction by svm
svmMOD<-train(lrn_ksvm,task)
predKSVM<-predict(svmMOD,task)
predKSVM.t<-cbind(data.frame(predKSVM)[,3],dset[,4:5])
colnames(predKSVM.t)<-c("Pred","x","y")

prd.abs.ksvm<-subset(predKSVM.t,predKSVM.t$Pred==0)
prd.sciurus.ksvm<-subset(predKSVM.t,predKSVM.t$Pred==1)
prd.grey.ksvm<-subset(predKSVM.t,predKSVM.t$Pred==2)

prd.abs.ksvm<-vect(prd.abs.ksvm, geom=c("x","y"))
prd.sciurus.ksvm<-vect(prd.sciurus.ksvm, geom=c("x","y"))
prd.grey.ksvm<-vect(prd.grey.ksvm, geom=c("x","y"))

plot(LCMCrop)

plot(prd.sciurus.ksvm, add=T, col='red')
plot(prd.grey.ksvm, add=T, col='blue') #the model predicted 0 grey squirrel
plot(prd.abs.ksvm, add=T)


#7.3. Model tuning
#spatial partitioning
tune_level = makeResampleDesc(method = "SpCV", iters = 5)
#specifying random parameter value search
ctrl = makeTuneControlRandom(maxit = 10L)

#7.3.1. rf tuning
getParamSet(lrnRF)

paramsRF <- makeParamSet(
  makeIntegerParam("mtry",lower = 1,upper = 8),
  makeIntegerParam("nodesize",lower = 1,upper = 10)
)

set.seed(11) #for reproduction
tuneRF = tuneParams (learner = lrnRF, 
                     task=task,
                     resampling = tune_level,
                     par.set = paramsRF,
                     control = ctrl, 
                     measures = mlr::acc,
                     show.info = TRUE)
#mtry=1; nodesize=7 : acc.test.mean=0.6056488

#prediction by tuned rf
rftuned.set<-setHyperPars(lrnRF, par.vals=tuneRF$x)
rftunedMOD<-train(rftuned.set,task)
pred.rftuned<-raster::predict(rftunedMOD,task)
pred.t.rftuned<-cbind(data.frame(pred.rftuned)[,3],dset[,4:5])
colnames(pred.t.rftuned)<-c("Pred","x","y")

prd.abs.rftuned<-subset(pred.t.rftuned,pred.t.rftuned$Pred==0)
prd.sciurus.rftuned<-subset(pred.t.rftuned,pred.t.rftuned$Pred==1)
prd.grey.rftuned<-subset(pred.t.rftuned,pred.t.rftuned$Pred==2)

prd.abs.rftuned<-vect(prd.abs.rftuned, geom=c("x","y"))
prd.sciurus.rftuned<-vect(prd.sciurus.rftuned, geom=c("x","y"))
prd.grey.rftuned<-vect(prd.grey.rftuned, geom=c("x","y"))

plot(LCMCrop)

plot(prd.sciurus.rftuned, add=T, col='red')
plot(prd.grey.rftuned, add=T, col='blue') #the model predicted 0 grey squirrel
plot(prd.abs.rftuned, add=T)

#7.3.2. svm tuning
paramsSVM = makeParamSet(
  makeNumericParam("C", lower = -12, upper = 15, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -15, upper = 6, trafo = function(x) 2^x)
)

set.seed(11)
tuneSVM = tuneParams(learner = lrn_ksvm, 
                     task=task,
                     resampling = tune_level,
                     par.set = paramsSVM,
                     control = ctrl, 
                     measures = mlr::acc,
                     show.info = TRUE)

#C=1.79e+03; sigma=0.00508 : acc.test.mean=0.7332316

#7.4. The best model for prediction
#SVM model with tuned parameters
svmMODtuned<-ksvm(Pres~.,data=dset, type="C-svc",
             kernel=rbfdot,kpar=list(sigma=tuneSVM$x$sigma),C=tuneSVM$x$C,prob.model=TRUE)

#make prediction and plot
predKSVMtuned<-raster::predict(svmMODtuned,dset,"response",task)
predKSVMtuned.t<-cbind(data.frame(predKSVMtuned),dset[,4:5])
colnames(predKSVMtuned.t)<-c("Pred","x","y")

prd.abs.ksvm<-subset(predKSVMtuned.t,predKSVMtuned.t$Pred==0)
prd.sciurus.ksvm<-subset(predKSVMtuned.t,predKSVMtuned.t$Pred==1)
prd.grey.ksvm<-subset(predKSVMtuned.t,predKSVMtuned.t$Pred==2)

prd.abs.ksvm<-vect(prd.abs.ksvm, geom=c("x","y"))
prd.sciurus.ksvm<-vect(prd.sciurus.ksvm, geom=c("x","y"))
prd.grey.ksvm<-vect(prd.grey.ksvm, geom=c("x","y"))

plot(LCMCrop)

plot(prd.sciurus.ksvm, add=T, col='red')
plot(prd.grey.ksvm, add=T, col='blue') #the model predicted 0 grey squirrel
plot(prd.abs.ksvm, add=T)


