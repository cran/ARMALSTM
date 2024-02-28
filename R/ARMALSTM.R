

#' @title Hybrid ARMA-LSTM Model for Time Series Forecasting

#' @description The linear ARMA model is fitted to the time series. The significant number of PACF values of ARMA residuals are considered as the lag. The LSTM model is fitted to the ARMA residuals setting the lag value as the time step. User needs to install keras, tensorflow and reticulate packages as the prerequisite to implement this package.

#' @param X A univariate time series data
#' @param p Order of AR
#' @param q Order of MA
#' @param arfima Whether to include arfima (0<d<0.5)
#' @param dist.model The distribution density to use for the innovation. The default distribution for the mean model used is "ged". Other choices can be obtained from the rugarch package.
#' @param out.sample A positive integer indicating the number of periods before the last to keep for out of sample forecasting. To be considered as test data.
#' @param LSTM.units Number of units in the LSTM layer
#' @param ACTIVATION.function Activation function
#' @param DROPOUT Dropout rate
#' @param Optimizer Optimizer used for optimization of the LSTM model
#' @param Epochs Number of epochs of the LSTM model
#' @param LSTM.loss Loss function
#' @param LSTM.metrics Metrics

#' @import rugarch tseries keras tensorflow reticulate
#' @return
#' \itemize{
#'   \item ARMA.fit: Parameters of the fitted ARMA model
#'   \item ARMA.fitted: Fitted values of the ARMA model
#'   \item ARMA.residual: Residual values of the ARMA model
#'   \item ARMA.forecast: Forecast values obtained from the ARMA model for the test data
#'   \item ARMA.residual.nonlinearity.test: BDS test results for the ARMA residuals
#'   \item LSTM.lag: Lag used for the LSTM model
#'   \item FINAL.fitted: Fitted values of the hybrid ARMA-LSTM model
#'   \item FINAL.residual: Residual values of the hybrid ARMA-LSTM model
#'   \item FINAL.forecast: Forecast values obtained from the hybrid ARMA-LSTM model for the test data
#'   \item ACCURACY.MATRIX: RMSE, MAE and MAPE of the train and test data
#' }
#' @export
#'
#' @usage ARMA.LSTM(X, p, q, arfima = FALSE, dist.model= "ged", out.sample, LSTM.units,
#' ACTIVATION.function = "tanh", DROPOUT = 0.2, Optimizer ="adam", Epochs = 100,
#' LSTM.loss = "mse", LSTM.metrics = "mae")
#' @examples
#' \donttest{
#'y<-c(5,9,1,6,4,9,7,3,5,6,1,8,6,7,3,8,6,4,7,5)
#'my.hybrid<-ARMA.LSTM(y, p=1, q=0, arfima=FALSE, dist.model = "ged",
#'out.sample=10, LSTM.units=50, ACTIVATION.function = "tanh",
#'DROPOUT = 0.2, Optimizer ="adam", Epochs = 10, LSTM.loss = "mse", LSTM.metrics = "mae")
#'}

#' @references
#' \itemize{
#' \item Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.

#' \item Granger, C. W., & Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. Journal of time series analysis, 1(1), 15-29.

#' \item Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

#' \item Rakshit, D., Paul, R. K., & Panwar, S. (2021). Asymmetric price volatility of onion in India. Indian Journal of Agricultural Economics, 76(2), 245-260.

#' \item Rakshit, D., Paul, R. K., Yeasin, M., Emam, W., Tashkandy, Y., & Chesneau, C. (2023). Modeling Asymmetric Volatility: A News Impact Curve Approach. Mathematics, 11(13), 2793.

#' \item Rakshit, D., Roy, A., Atta, K., Adhikary, S., & Vishwanath. (2022). Modeling Temporal Variation of Particulate Matter Concentration at Three Different Locations of Delhi. International Journal of Environment and Climate Change, 12(11), 1831-1839.

#' }





ARMA.LSTM<- function (X, p, q, arfima=FALSE, dist.model = "ged", out.sample, LSTM.units,
                      ACTIVATION.function = "tanh", DROPOUT = 0.2, Optimizer ="adam",
                      Epochs = 100, LSTM.loss = "mse", LSTM.metrics = "mae")
{

  arma.spec<-rugarch::arfimaspec(mean.model = list(armaOrder = c(p,q), include.mean = TRUE,
                                                   arfima = arfima, external.regressors = NULL),
                                 distribution.model = dist.model)

  arma.fit<-rugarch::arfimafit(spec=arma.spec, data=X, out.sample = out.sample)

  arma.fitted<-arma.fit@fit$fitted.values # FITTED VALUES
  arma.residual<-arma.fit@fit$residuals # residual values

  arma.forecast.temp<-rugarch::arfimaforecast(arma.fit, n.ahead=out.sample) #FORECAST
  arma.forecast<-rugarch::fitted(arma.forecast.temp)
  arma.forecast<-as.data.frame(arma.forecast)


  # test for nonlinearity of ARMA residual
  nonlinearity.test<- tseries::bds.test(arma.residual)

  # PACF of ARMA residual
  pacf.res<-stats::pacf(arma.residual)



  significant_pacf <- which(abs(pacf.res$acf) > (stats::qnorm(0.975)) / sqrt(length(X)))

  lag.lstm <- NULL
  for (i in 1:(length(significant_pacf)-1)){
    if(significant_pacf[i] == significant_pacf[i+1]-1){
      lag.lstm <- significant_pacf[i+1]
    }
  }


  ####### LSTM starts here #################

  data_LSTM <- arma.residual

  # Function to create lagged data matrix
  create_lagged_matrix <- function(data, lag) {
    n <- length(data)
    matrix_data <- matrix(NA, nrow = n - lag , ncol = lag + 1)
    for (i in 1:(n - lag)) {
      matrix_data[i, ] <- data[(i + lag):i]
    }
    return(matrix_data)
  }

  # Create lagged matrix
  lagged_matrix <- create_lagged_matrix(data_LSTM, lag.lstm)



  LSTM.feature <- lagged_matrix[, -1]
  LSTM.target <- lagged_matrix[, 1]
  LSTM.feature<- array(LSTM.feature, dim = c(nrow(LSTM.feature), lag.lstm, 1))

  # LSTM model
  lstm_model <- keras::keras_model_sequential() %>%
    layer_lstm(units =LSTM.units, input_shape = c(lag.lstm, 1), activation=ACTIVATION.function, dropout=DROPOUT) %>%
    layer_dense(units = 1)

  lstm_model %>% compile(optimizer = Optimizer, loss= LSTM.loss, metrics=LSTM.metrics)

  summary(lstm_model)

  history<- lstm_model %>% fit(
    LSTM.feature,
    LSTM.target,
    batch_size = 1,
    epochs = Epochs)

  lstm.fitted <- lstm_model %>%  stats::predict(LSTM.feature)

  #### LSTM forecast ###########################################
  whole_feature <- lagged_matrix[, -1]
  forecast.inputdata <- whole_feature[nrow(whole_feature), ]


  interim.forecast <- NULL
  n.test <- out.sample
  for (i in 1: (n.test+1)){
    pred <- lstm_model %>% stats::predict(array(forecast.inputdata, dim = c(1, lag.lstm, 1)))
    interim.forecast<- c(interim.forecast, pred)
    forecast.inputdata <- c(forecast.inputdata, pred)
    forecast.inputdata <- forecast.inputdata[-1]
  }



  lstm.forecast <- interim.forecast[-1]

  #########################  hybrid model output ########################
  final.fitted<- arma.fitted[(lag.lstm+1):length(arma.fitted)]+lstm.fitted
  final.forecast<- arma.forecast[,1]+lstm.forecast
  final.residual<- X[(lag.lstm+1):length(arma.fitted)]-final.fitted



  ############# accuracy measures ######################
  Accuracy.matrix<- matrix(nrow=2, ncol=3)
  row.names(Accuracy.matrix)<-c("Train", "Test")
  colnames(Accuracy.matrix)<-c("RMSE", "MAE", "MAPE")

  train.original<-X[(lag.lstm+1):length(arma.fitted)]

  Accuracy.matrix[1,1]<-round(sqrt(mean((train.original-final.fitted)^2)), digits = 4)
  Accuracy.matrix[1,2]<-round(mean(abs(train.original-final.fitted)), digits = 4)
  Accuracy.matrix[1,3]<-round(mean(abs((train.original-final.fitted)/train.original))*100, digits = 4)

  test.original<-X[(length(X)-n.test+1):length(X)]

  Accuracy.matrix[2,1]<-round(sqrt(mean((test.original-final.forecast)^2)), digits = 4)
  Accuracy.matrix[2,2]<-round(mean(abs(test.original-final.forecast)), digits = 4)
  Accuracy.matrix[2,3]<-round(mean(abs((test.original-final.forecast)/test.original))*100, digits = 4)


  output<- list(ARMA.fit = arma.fit,
                ARMA.fitted = arma.fitted,
                ARMA.residual = arma.residual,
                ARMA.forecast = arma.forecast[,1],
                ARMA.residual.nonlinearity.test = nonlinearity.test,
                LSTM.lag = lag.lstm,
                FINAL.fitted = final.fitted,
                FINAL.residual = final.residual,
                FINAL.forecast = final.forecast,
                ACCURACY.MATRIX = Accuracy.matrix)

  return(output)

}



