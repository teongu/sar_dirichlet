require("classInt") # discretize numeric variable
require("compositions") # compositional data
require("dplyr") # dplyr data
require("ggplot2") # ggplot functions
require("isoband") # Joint distribution 
require("Matrix") # sparse matrix
require("rgdal") # import spatial data
require("sp") # spatial data
require("sf") # spatial data V2
require("spdep") # spatial econometric modelling


URL <- "http://www.thibault.laurent.free.fr/code/GLT/"
fil <- "contours.zip"
if (!file.exists(fil)) download.file(paste(URL, fil, sep = ""), fil)
unzip(fil)

mapMAP <- readOGR(dsn = "contours", layer = "ADTCAN_region")
mapMAP@data$CODE <- as.numeric(as.character(mapMAP@data$CODE))
n <- nrow(mapMAP)
coords <- coordinates(mapMAP)




W1.listw <- nb2listw(knn2nb(knearneigh(coords, 10)), 
                     style = "W")
W_simu <- listw2mat(W1.listw)


load(url("http://www.thibault.laurent.free.fr/code/spatial_coda/R/data_cantons.RData"))

source(url("http://www.thibault.laurent.free.fr/code/spatial_coda/R/preparation_base.R"))

Ye <- as(y_ilr, "matrix")

Xe <- as(cbind(1, x2_df[, c("diplome3_ilr1", "diplome3_ilr2",
                            "employ_ilr1", "employ_ilr2", "employ_ilr3", 
                            "employ_ilr4",
                            "age3_ilr1", "age3_ilr2", 
                            "unemp_rate", "income_rate", "voters")]),
         "matrix")
ne <- nrow(Xe)
k <- ncol(Xe)


library(MASS)

coords_fr <- st_coordinates(st_centroid(contours_occitanie_no0))
W_listw <- nb2listw(knn2nb(knearneigh(coords_fr[, 1:2], 10)), 
                    style = "W")
W_dep <- listw2mat(W_listw)

write.matrix(W_dep,file="W_elections_10nn.csv")

write.matrix(contours_occitanie_no0,file="contours_occitanie.csv", sep=',')

Y_occitanie = matrix(cbind(contours_occitanie_no0$percent_left,
             contours_occitanie_no0$percent_right,
             contours_occitanie_no0$percent_others),ncol=3)
write.matrix(Y_occitanie,file="occitanie/Y_occitanie.csv", sep=';')


X_occitanie = matrix(cbind(contours_occitanie_no0$dep_canton,
                           contours_occitanie_no0$POP,
                           contours_occitanie_no0$age_mineur,
                           contours_occitanie_no0$age_1824,
                           contours_occitanie_no0$age_2540,
                           contours_occitanie_no0$age_4055,
                           contours_occitanie_no0$age_5564,
                           contours_occitanie_no0$age_65,
                           contours_occitanie_no0$PIMP13,
                           contours_occitanie_no0$MED13,
                           contours_occitanie_no0$NBMENFISC13,
                           contours_occitanie_no0$P14_CHOM1564,
                           contours_occitanie_no0$P14_ACT1564,
                           contours_occitanie_no0$P14_EMPLT,
                           contours_occitanie_no0$P09_EMPLT,
                           contours_occitanie_no0$P14_RP_PROP,
                           contours_occitanie_no0$P14_RP,
                           contours_occitanie_no0$no_diplom,
                           contours_occitanie_no0$capbep,
                           contours_occitanie_no0$bac,
                           contours_occitanie_no0$diplom_sup,
                           contours_occitanie_no0$french,
                           contours_occitanie_no0$foreign,
                           contours_occitanie_no0$unemp_rate,
                           contours_occitanie_no0$employ_evol,
                           contours_occitanie_no0$owner_rate,
                           contours_occitanie_no0$income_rate), nrow=207)

X_occitanie = matrix(cbind(contours_occitanie_no0$dep_canton,
                           contours_occitanie_no0$POP,
                           contours_occitanie_no0$AZ,
                           contours_occitanie_no0$BE,
                           contours_occitanie_no0$FZ,
                           contours_occitanie_no0$GU,
                           contours_occitanie_no0$OQ,
                           contours_occitanie_no0$age_mineur,
                           contours_occitanie_no0$age_1824,
                           contours_occitanie_no0$age_2540,
                           contours_occitanie_no0$age_4055,
                           contours_occitanie_no0$age_5564,
                           contours_occitanie_no0$age_65,
                           contours_occitanie_no0$no_diplom,
                           contours_occitanie_no0$capbep,
                           contours_occitanie_no0$bac,
                           contours_occitanie_no0$diplom_sup,
                           contours_occitanie_no0$french,
                           contours_occitanie_no0$foreign,
                           contours_occitanie_no0$unemp_rate,
                           contours_occitanie_no0$owner_rate), nrow=207)


colnames(X_occitanie) <- c("dep_canton", "POP", "AZ", "BE", "FZ", "GU", "OQ",
                           "age_mineur", "age_1824",
                 "age_2540", "age_4055", "age_5564", "age_65",
                 "no_diplom", "capbep", "bac", "diplom_sup",
                 "french", "foreign", "unemp_rate", "owner_rate")



write.matrix(X_occitanie,file="occitanie/X_occitanie_bis.csv", sep=';')

write.matrix(coords_fr,file="occitanie/coordinates_cendroids.csv", sep=";")
