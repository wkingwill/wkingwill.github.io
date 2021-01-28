#library
install.packages("Hmisc")
install.packages("kableExtra")
library(DataExplorer)
library(skimr)
library(data.table)
library(data.table)
library(dplyr)
library(ggplot2)
library(e1071)
library(randomForest)
library(doParalle)
library(Hmisc)  
library(outlier)
library(knitr)
library(kableExtra)
library(leaflet)

help()
       
        
#Create folder path
folder_path <- "C:/Users/Eric/OneDrive/IE/R/Project"

#load data sets
solar_dataset <- readRDS(file.path(folder_path, "solar_dataset.RData"));
additional_variables <- readRDS(file.path(folder_path, "additional_variables.RData"));
station_info <- fread(file.path(folder_path, "station_info.csv"))

# split data set  
train1 <- head(solar_dataset,5113)
pred_test <- solar_dataset[5114:6909,]


##Prelimerinary view of datasets
head(solar_dataset)
head(station_info)
head(additional_variables)

# STAT SUMMARY OF TABLES
sapply(train1, summary)
sapply(additional_variables, summary)

#For html summary
Train_summary <- as.data.frame(sapply(train1[ ,-1], summary))
kable(Train_summary,format = "html", caption = "Percentage Null values for additional_variables", ggtheme = theme_classic())%>%
  kable_styling() %>%
  scroll_box(width = "600px",  height = "400px") 

additional_summary <- as.data.frame(sapply(additional_variables[ ,-1], summary))
kable(additional_summary,format = "html", caption = "Percentage Null values for additional_variables", ggtheme = theme_classic())%>%
  kable_styling() %>%
  scroll_box(width = "600px",  height = "400px") 



#plot basic description of data set
plot_intro(solar_dataset, title = "Intro for solar_dataset")
plot_intro(train1, title = "Intro for train1")

plot_intro(station_info, title = "Intro for station_info")
plot_intro(additional_variables, title = "Intro for additional_variables")
help(plot_intro)


### Check for missing values 
sapply(train1, function(x){sum(is.na(x))});
sapply(solar_dataset, function(x){sum(is.na(x))});
sapply(station_info, function(x){sum(is.na(x))});
sapply(additional_variables, function(x){sum(is.na(x))});



additional_null <- as.data.frame(sapply(additional_variables, function(x){sum(is.na(x))}));
names(additional_null) <- "No. Null Values"




## can be seen that only missing values are in the additional variables datasets
## therefore no null value inputation needs to be done on the other data sets


### check frequency of nulls in columns of additional variables data set 

sapply(additional_variables, function(x){100*sum(is.na(x))/length(x)});
null_feq <- sapply(additional_variables, function(x){100*sum(is.na(x))/length(x)});
## no columns with large percentage nulls so keep all columns

###Fill missing values with mean
fill_missing_values <- function(x){
  if (class(x) == "numeric" ){
    x[is.na(x)] <- mean(x, na.rm = TRUE); # Replace with mean
  } else {
    
  }
  return(x);
}

additional_variables <- data.frame(sapply(additional_variables, fill_missing_values));
sapply(additional_variables, function(x){sum(is.na(x))});
additional_pred <- additional_variables


#plot Histogram of continous variables 
plot_histogram(solar_dataset) ## All columns
help("plot_histogram")

# plot histogram of a few variables to get idea of distributions.
df_solar <- as.data.frame(solar_dataset)
plot_histogram(df_solar[, c("ADAX","BUFF","GUTH", "HOOK", "STUA", "SULP","PC1","PC10","PC40","PC80", "PC120","PC160", "PC200", "PC240","PC280","PC300", "PC340", "PC350")], nrow = 3,ncol = 3, ggtheme = theme_classic())  


### Plot box plot 
par(mfrow=c(2,2))
boxplot(train1$ACME, main = "ACME")
boxplot(train1$BUFF, main = "BUFF")
boxplot(train1$GUTH, main = "GUTH")
boxplot(train1$HOOK, main = "HOOK")
boxplot(train1$STUA, main = "STUA")
boxplot(train1$SULP, main = "SULP")
boxplot(train1$PC1, main = "PC1")
boxplot(train1$PC10, main = "PC10")
boxplot(train1$PC40, main = "PC40")
boxplot(train1$PC80, main = "PC80")
boxplot(train1$PC120, main = "PC120")
boxplot(train1$PC160, main = "PC160")
boxplot(train1$PC200, main = "PC200")
boxplot(train1$PC240, main = "PC240")
boxplot(train1$PC280, main = "PC280")
boxplot(train1$PC300, main = "PC300")
boxplot(train1$PC340, main = "PC340")
boxplot(train1$PC350, main = "PC350")


### Plot of average solar production over time

dt_train1_solar_avg <- as.data.table(train1[,1:99])
dt_train1_solar_avg$AvgSolar <- rowMeans(dt_train1_solar_avg[ ,2:99])
dt_train1_solar_avg <- dt_train1_solar_avg[, c("Date", "AvgSolar")]

dt_train1_solar_avg$Date <- as.Date(dt_train1_solar_avg$Date, format = "%Y%m%d")


solar_avg_plot <- ggplot(dt_train1_solar_avg, aes(x = Date, y = AvgSolar))+
  geom_line() +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") 
  
  solar_avg_plot <- solar_avg_plot +   scale_colour_brewer(palette = "YlOrRd")
  print(solar_avg_plot)

 
 # AvgSolar_dataset <- cbind(dt_train1_solar_avg$AvgSolar, train1[, 99:456])

  

  
### Create map of all the Mesonet Sites with labels
stations <- as.data.frame(station_info)

m = leaflet() %>% addTiles() %>% addMarkers(lng = stations$elon, lat = stations$nlat, label = stations$stid)
m




## Check correltions and p values - just using pearson correlations

Corr_res <- rcorr(as.matrix(train1))
p_value_index <-Corr_res$P < 0.05)


