#Jawad KASSIMI
# Machine Learning Project : Dynamic portfolio backtesting project

library(shiny)
library(shinydashboard)
library(shinyWidgets)
library(plotly)
library(DT)
library(tidyverse)
library(ranger)
library(zoo)
library(MASS)
library(dplyr)

# Here I load the data provided
load(file.path(getwd(), "spx_macro.RData"))

# I chose to keep only the stocks that are present at all dates for consistency
spx_balanced <- spx_macro %>%
  group_by(symbol) %>% mutate(n = n()) %>%
  ungroup() %>% mutate(max_n = max(n)) %>%
  filter(n == max_n) %>% dplyr::select(-n, -max_n)

# I clean the data by dropping rows with missing key features
spx_clean <- spx_balanced %>%
  drop_na(fwd_return, pb, d2e, profit_margin, mkt_cap, vol_1y)

#Identifying which numeric features must be scaled
features_to_scale <- spx_clean %>%
  dplyr::select(where(is.numeric)) %>%
  dplyr::select(-fwd_return) %>% names()

spx_scaled <- spx_clean %>%
  group_by(date) %>%
  mutate(across(all_of(features_to_scale), ~ as.numeric(scale(.x)))) %>%
  ungroup()

# I add lagged and ratio features to reflect financial signals
spx_features <- spx_scaled %>%
  arrange(symbol, date) %>%
  group_by(symbol) %>%
  mutate(
    mom_1m = lag(return, 1),
    mom_3m = (lag(return, 1) + lag(return, 2) + lag(return, 3))/3,
    vol_trend = vol_1y / lag(vol_1y, 3),
    size_mom = mkt_cap / lag(mkt_cap, 3),
    pb_trend = pb / lag(pb, 3),
    d2e_trend = d2e / lag(d2e, 3),
    profit_trend = profit_margin / lag(profit_margin, 3)
  ) %>%
  ungroup() %>%
  mutate(
    across(c(vol_trend, size_mom, pb_trend, d2e_trend, profit_trend), ~ ifelse(is.na(.), 1, .)),
    across(c(mom_1m, mom_3m), ~ ifelse(is.na(.), 0, .))
  ) %>%
  filter(!is.na(fwd_return))

# I prepare the initial train/test split to initialize my feature list
dates <- unique(spx_features$date)
split_date <- dates[round(length(dates) * 0.5)]
train_data <- spx_features %>% filter(date <= split_date)
test_data <- spx_features %>% filter(date > split_date)

# Here I select the final features used for my model 
model_features <- c("mom_1m", "mom_3m", "vol_1y", "vol_trend", 
                    "pb", "pb_trend", "d2e", "d2e_trend", 
                    "profit_margin", "profit_trend", 
                    "mkt_cap", "size_mom", "dy", "return")
model_features <- model_features[model_features %in% names(train_data)]


emlyon_red <- "#d90429"
emlyon_white <- "#ffffff"
emlyon_grey <- "#f8f9fa"

# UI
#Preparing the dashboard and the sidebar
ui <- dashboardPage(
  skin = "red",
  dashboardHeader(title = NULL),
  dashboardSidebar(div(
    style = "text-align: center; font-weight: bold; font-size: 20px; color: white; padding: 15px 0;",
    "Machine Learning Project"
  ),
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("chart-line")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    ),
    br(),
    h4("Parameters", align = "center", style = "color: white"),
    dateRangeInput("backtest_range", "Backtest Period:",
                   start = min("2010-01-31"),
                   end = max("2020-12-31"),
                   min = min(spx_features$date),
                   max = max(spx_features$date)),
    dateInput("train_start", "Training Start Date:",
              value = min("2005-01-31"),
              min = min(spx_features$date),
              max = max(spx_features$date)),
    sliderInput("num_trees", "Number of Trees:", min = 5, max = 500, value = 100, step = 5),
    actionBttn("run", "Start Backtest", style = "fill", color = "danger", icon = icon("play"), block = TRUE),
    br(),
    helpText("The computations can take few seconds.", style = "color:white;text-align:center;")
  ),
  dashboardBody(
    tags$head(tags$style(HTML(".skin-red .main-header .logo { background-color: #d90429; } 
                               .skin-red .main-header .navbar { background-color: #d90429; }
                               .skin-red .main-sidebar { background-color: #2f2f2f; overflow-y: visible; z-index: 9999; }
                               .sidebar { overflow-y: visible !important; z-index: 9999; }
                               .datepicker.dropdown-menu {z-index: 99999 !important;}  
                               .main-header .sidebar-toggle { margin-left: 20px !important; }"))),
    tabItems(
      tabItem(tabName = "dashboard",
              fluidRow(
                tabBox(width = 12,
                       tabPanel("Project Presentation",
                                fluidRow(
                                  box(title = "Introduction", width = 12, status = "danger", solidHeader = TRUE,
                                      tags$p("For this project, I decided to work on project type 4.2: Dynamic Portfolio Backtesting, because I wanted to explore how machine learning could be used not just for prediction, but for actual portfolio construction.While the implementation is the result of personal work, it draws inspiration from online resources and artificial intelligence tools to guide me for the development of this project."),
                                      tags$p("My goal was to build a time-evolving investment strategy, where both the signal and the portfolio weights are updated each month based on fresh data. I chose to work with the main dataset provided (covering ~500 US stocks and macroeconomic indicators from 1991 to 2025)."),
                                      tags$p("This Shiny dashboard allows me to interactively test, visualize, and reflect on the behavior of my model-driven strategy.Unlike my previous project, which was delivered as a static HTML notebook, this time I wanted to explore the possibilities offered by Shiny. I found the idea of being able to interactively modify the parameters,such as backtest period, training window, or model complexity, particularly appealing, as it brings the strategy closer to how it would be used in a real-world setting.")
                                  ),
                                  
                                  box(title = "Machine Learning Signal and Portfolio", width = 12, status = "danger", solidHeader = TRUE,
                                      tags$p("For this project, I chose to use a Random Forest model, implemented with the {ranger} package in R. I appreciate this algorithm for its robustness and ability to handle non-linear patterns without requiring too much tuning. It also gives me access to feature importance scores, which helps me interpret the drivers behind the model’s predictions."),
                                      
                                      tags$p("The target I wanted to predict is the one-month forward return (`fwd_return`) of each stock.Every month, I retrain the model using all available past data starting from a chosen training date. To feed the model, I engineered a set of features capturing different investment styles: momentum, value, size, volatility, leverage, and profitability, all computed with meaningful lags to reflect realistic investor behavior."),
                                      
                                      tags$p("Once predictions are made, I rank all the stocks and select the top 20% with the highest predicted returns. These form my long-only portfolio for the month. Rather than equal-weighting, I assign weights proportionally to the predicted returns. This makes the strategy responsive: stocks with stronger signals have a greater influence in the portfolio."),
                                      
                                      tags$p("By retraining the model and reshaping the portfolio every month, I ensure that the signal adapts to changing market conditions. I also take care to strictly separate training and test data, to avoid any look-ahead bias.")
                                  ),
                                  
                                  box(title = "Metrics Overview", width = 12, status = "danger", solidHeader = TRUE,
                                      tags$ul(
                                        tags$li(tags$b("Cumulative Return:"), " tracks the evolution of wealth if you had invested following the strategy vs. the market."),
                                        tags$li(tags$b("Monthly Return:"), " shows the monthly performance of the strategy."),
                                        tags$li(tags$b("Sharpe Ratio:"), " computed as the 12-month rolling Sharpe, helps evaluate risk-adjusted return."),
                                        tags$li(tags$b("Return Distribution:"), " compares the distribution of returns to a normal and a Student-t law."),
                                        tags$li(tags$b("Prediction Error:"), " plots predicted vs actual returns and displays the RMSE."),
                                        tags$li(tags$b("Top Stocks:"), " shows which stocks were most often selected."),
                                        tags$li(tags$b("Feature Importance:"), " reflects which predictors were most used by the Random Forest."),
                                        tags$li(tags$b("Performance Metrics Table:"), " summarizes annualized return, volatility, Sharpe ratio, drawdowns, and VaR."),
                                        tags$ul(
                                          tags$li(tags$b("Sharpe Ratio:"), " measures risk-adjusted return (mean divided by standard deviation of returns)."),
                                          tags$li(tags$b("Max Drawdown:"), " shows the worst peak-to-trough decline in portfolio value."),
                                          tags$li(tags$b("MAR Ratio:"), " is the ratio of annualized return to maximum drawdown, rewarding stable performance."),
                                          tags$li(tags$b("VaR 95%:"), " estimates the potential portfolio loss in a worst-case scenario with 95% confidence.")
                                        )
                                      )
                                  ),
                                  box(title = "Reference Test", status = "danger", solidHeader = TRUE, width = 12,
                                      tags$p("To evaluate my strategy in a concrete and reproducible way, I ran a benchmark test using the following parameters:"),
                                      tags$ul(
                                        tags$li("Backtest period: from January 2000 to December 2023"),
                                        tags$li("Training start date: January 1995"),
                                        tags$li("Number of trees: 100 (Random Forest)")
                                      ),
                                      tags$p("Just to give an idea about the computation time, It took me about 11 minutes to run the backtest with those parameters and I have the old Intel MacBook Pro computer. The strategy was trained and re-estimated dynamically each month during this 24-year period. As shown in the 'Cumulative Return' chart, my portfolio outperformed the market significantly over time, with an annualized return of 17,94%, especially after 2018. This suggests that the machine learning signal (predicted forward returns) effectively captured persistent patterns in the cross-section of stock returns."),
                                      tags$p("In terms of risk-adjusted performance, the Sharpe Ratio stabilized around 0.60, which is a solid result for an equity strategy. The 'Monthly Return' plot confirms that most months delivered modest returns, with a few large outliers. The 'Return Distribution' shows that the empirical returns are better fitted by a Student-t distribution than by a normal law, which justifies the use of more robust error metrics."),
                                      tags$p("The RMSE of prediction (Root Mean Squared Error) was around 0.104, confirming that the model captures meaningful information while still leaving room for improvement."),
                                      tags$p("From a feature perspective, momentum and price-to-book dynamics appear as the most influential factors. The top stocks most often selected across the backtest include names like AMD, TPL, and ADSK, which highlights the model's tendency to favor consistent outperformers."),
                                      tags$p("This reference test helped me confirm that the strategy works as intended, and sets a baseline for further sensitivity analysis or strategy improvements.")
                                  ),
                                  box(title = "Conclusion", status = "danger", solidHeader = TRUE, width = 12,
                                      tags$p("This project allowed me to design and implement a dynamic investment strategy driven by a machine learning model and visualized in R through a Shiny interface."),
                                      tags$p("By working on project type 4.2: Dynamic Portfolio Backtesting, I had the opportunity to go beyond traditional static models and explore how reallocation based on predictive signals can generate value."),
                                      tags$p("I particularly appreciated the interactive nature of the dashboard, which allows the user to change key parameters (dates, training period, number of trees) and instantly observe the impact on performance."),
                                      tags$p("This work also helped me better understand the role of financial features in model training, the importance of avoiding look-ahead bias, and the need for robust validation to interpret results meaningfully.")
                                  ),
                                )),
                       tabPanel("Cumulative Return",tags$p("I use this chart to compare the cumulative performance of my ML-based strategy against a naive equal-weighted benchmark.The curve allows me to assess whether the model adds value over time. A growing gap in favor of the ML strategy supports its relevance."),
                                plotlyOutput("cumulativePlot", height = "400px")),
                       tabPanel("Monthly Return", tags$p("This bar chart shows the monthly returns of the ML-based portfolio. I use it to identify periods of strong performance or significant drawdowns, and to assess the volatility of monthly outcomes over time."),
                                plotlyOutput("monthlyPlot", height = "400px")),
                       tabPanel("Sharpe Ratio",tags$p("Here I display the 12-month rolling Sharpe ratio to evaluate the risk-adjusted return of my strategy.A consistently high Sharpe indicates that the model provides returns in excess of risk in a stable way."),
                                plotlyOutput("sharpePlot", height = "400px")),
                       tabPanel("Return Distribution",tags$p("This density plot compares the empirical distribution of the strategy’s returns to theoretical normal and Student-t distributions.I include it to check whether the returns are skewed or heavy-tailed, which could, for instance, impact the risk management."),
                                plotlyOutput("distributionPlot", height = "400px")),
                       tabPanel("Prediction Error",fluidRow(valueBoxOutput("rmseBox", width = 12)),
                                fluidRow( tags$p("In this scatterplot, I compare my model’s predicted returns to the actual future returns (fwd_return).Ideally, the points should lie close to the dashed line (perfect prediction).This chart helps me understand how accurate my model is and whether it tends to systematically over- or under-estimate returns."),
                                         plotlyOutput("predictionPlot", height = "400px")
                                ))
                )
              ),
              fluidRow(
                box(title = "Top Stocks", width = 6, status = "danger", solidHeader = TRUE, plotlyOutput("selectionPlot")),
                box(title = "Feature Importance", width = 6, status = "danger", solidHeader = TRUE, plotlyOutput("importancePlot"))
              ),
              fluidRow(
                box(title = "Performance Metrics", width = 12, status = "danger", solidHeader = TRUE, DTOutput("performanceTable"))
              )
      ),
      tabItem(tabName = "about",
              box(title = "About", width = 12, status = "danger", solidHeader = TRUE,
                  tags$div(
                    style = "padding: 20px; font-size: 16px;",
                    tags$p(strong("Author:"), " Jawad KASSIMI"),
                    tags$p(strong("Program:"), " MSc in Finance – emlyon business school"),
                    tags$p(strong("Course:"), " Machine Learning in Factor Investing"),
                    tags$p(strong("Professor:"), " Guillaume Coqueret"),
                    tags$p(strong("Project Type:"), " 4.2: Dynamic Portfolio Backtesting"),
                    tags$hr()
                  )
              )
      )
      
    )
  )
)

# Server 
server <- function(input, output, session) {
  results <- eventReactive(input$run, {
    withProgress(message = "Running backtest...", value = 0, {
      start_bt <- input$backtest_range[1] # I retrieve the selected parameters from the UI
      end_bt <- input$backtest_range[2]
      train_start <- input$train_start
      num_trees <- input$num_trees
      
      dates_test <- unique(test_data$date)
      dates_test <- dates_test[dates_test >= start_bt & dates_test <= end_bt]
      
      if (length(dates_test) == 0) {
        showNotification("No dates in selected backtest range. Adjust the parameters.", type = "error")
        return(NULL)
      }
     
      portfolios <- data.frame()  # I initialize empty structures to collect results across time
      importance_matrix <- matrix(0, nrow = length(model_features), ncol = 0)
      rownames(importance_matrix) <- model_features
      
      # I create a loop through each backtest date and retrain the model dynamically
      for (current_date in dates_test) {
        incProgress(1/length(dates_test), detail = paste("Date", current_date))
        current_train <- test_data %>% filter(date >= train_start & date < current_date) # I create a rolling training window and isolate the test month
        current_test <- test_data %>% filter(date == current_date)
        if (nrow(current_test) == 0 || nrow(current_train) < 50) next
        
        # I train a Random Forest on past data
        rf_model <- ranger(
          formula = paste("fwd_return", "~", paste(model_features, collapse = "+")),
          data = current_train,
          num.trees = num_trees,
          mtry = min(floor(length(model_features)/3), 5),
          min.node.size = 10,
          importance = "impurity",
          seed = 123
        )
        
        importance_matrix <- cbind(importance_matrix, rf_model$variable.importance[model_features])
        current_test$predicted_return <- predict(rf_model, data = current_test)$predictions  # It predict forward returns on the current month
        
        # I build the portfolio by selecting the top quantile and weighting by predicted return
        current_portfolio <- current_test %>%
          mutate(quantile = ntile(predicted_return, 5)) %>%
          filter(quantile == 5) %>%
          mutate(weight = predicted_return / sum(predicted_return))
        
        portfolios <- bind_rows(portfolios, current_portfolio)
      }
     
      all_predictions <- portfolios %>%
        dplyr::select(date, symbol, fwd_return, predicted_return)
      
      perf_portfolio <- portfolios %>% # I compute the portfolio return
        group_by(date) %>%
        summarize(portfolio_return = sum(fwd_return * weight), .groups = 'drop')
      
      perf_market <- test_data %>% # I also compute a market benchmark equal-weighted to compare
        group_by(date) %>%
        summarize(market_return = mean(fwd_return), .groups = 'drop')
      
      perf_comparison <- left_join(perf_portfolio, perf_market, by = "date") %>%
        arrange(date) %>%
        mutate(
          cum_portfolio = cumprod(1 + portfolio_return),
          cum_market = cumprod(1 + market_return)
        )
      
      top_stocks <- portfolios %>% # Here I identify which stocks were most frequently selected to plot them
        group_by(symbol) %>%
        summarize(selection_count = n(), .groups = 'drop') %>%
        arrange(desc(selection_count)) %>%
        slice_max(selection_count, n = 10)
      # Computing the average feature importances and RMSE
      if (ncol(importance_matrix) == 0) {
        importance <- rep(0, length(model_features))
        names(importance) <- model_features
      } else {
        importance <- rowMeans(importance_matrix, na.rm = TRUE)
        rmse <- sqrt(mean((all_predictions$predicted_return - all_predictions$fwd_return)^2, na.rm = TRUE))
      }
      
      list(
        perf = perf_comparison,
        returns = perf_portfolio$portfolio_return,
        top = top_stocks,
        importance = importance,
        predictions = all_predictions,
        rmse = rmse
      )
    })
  })
  
  #Here are all the plots I display in my Shiny App (cumulative/monthly performance,sharpe ratio,return distribution etc)
  output$cumulativePlot <- renderPlotly({
    req(results())
    df <- results()$perf %>%
      pivot_longer(cols = starts_with("cum"), names_to = "strategy", values_to = "cum_return")
    p <- ggplot(df, aes(x = date, y = cum_return, color = strategy)) +
      geom_line(size = 1) +
      labs(title = "Cumulative Performance", x = "Date", y = "Cumulative Return") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$monthlyPlot <- renderPlotly({
    req(results())
    p <- ggplot(results()$perf, aes(x = date, y = portfolio_return)) +
      geom_col(fill = "#d90429") +
      labs(title = "Monthly Portfolio Returns", x = "Date", y = "Monthly Return") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$sharpePlot <- renderPlotly({
    req(results())
    df <- results()$perf %>%
      mutate(roll_sharpe = zoo::rollapply(portfolio_return, width = 12, FUN = function(x) {
        if (length(x) < 12) return(NA)
        mean(x) / sd(x)
      }, fill = NA, align = "right"))
    p <- ggplot(df, aes(x = date, y = roll_sharpe)) +
      geom_line(color = "darkred", size = 1) +
      labs(title = "Rolling 12-Month Sharpe Ratio", x = "Date", y = "Sharpe") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$distributionPlot <- renderPlotly({
    req(results())
    returns <- results()$returns
    mu <- mean(returns, na.rm = TRUE)
    sigma <- sd(returns, na.rm = TRUE)
    fit_t <- fitdistr(returns, densfun = "t")
    df_t <- fit_t$estimate["df"]
    mu_t <- fit_t$estimate["m"]
    sigma_t <- fit_t$estimate["s"]
    x_vals <- seq(min(returns), max(returns), length.out = 300)
    dens_df <- tibble(
      x = x_vals,
      normal = dnorm(x_vals, mu, sigma),
      student = dt((x_vals - mu_t) / sigma_t, df_t) / sigma_t
    )
    p <- ggplot(data.frame(Returns = returns), aes(x = Returns)) +
      geom_histogram(aes(y = ..density..), bins = 30, fill = "#ff4d4d", alpha = 0.6) +
      geom_line(data = dens_df, aes(x = x, y = normal), color = "darkgreen", linetype = "dashed") +
      geom_line(data = dens_df, aes(x = x, y = student), color = "black") +
      labs(title = "Return Distribution: Normal vs Student-t", x = "Return", y = "Density") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$selectionPlot <- renderPlotly({
    req(results())
    p <- ggplot(results()$top, aes(x = reorder(symbol, selection_count), y = selection_count)) +
      geom_col(fill = "#d90429") +
      coord_flip() +
      labs(title = "Top 10 Most Selected Stocks", x = "Symbol", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$importancePlot <- renderPlotly({
    req(results())
    importance_df <- data.frame(
      Feature = names(results()$importance),
      Importance = as.numeric(results()$importance)
    ) %>%
      arrange(desc(Importance)) %>%
      mutate(Feature = fct_reorder(Feature, Importance))
    p <- ggplot(importance_df, aes(x = Importance, y = Feature)) +
      geom_col(fill = "steelblue") +
      labs(title = "Feature Importance (Random Forest)", x = "Importance Score", y = "Feature") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$performanceTable <- renderDT({
    req(results())
    ret <- results()$returns
    ann_factor <- 12
    mu <- mean(ret, na.rm = TRUE) * ann_factor
    vol <- sd(ret, na.rm = TRUE) * sqrt(ann_factor)
    sharpe <- mu / vol
    cum <- cumprod(1 + ret)
    dd <- min(cum / cummax(cum) - 1, na.rm = TRUE)
    mar <- mu / abs(dd)
    var_95 <- quantile(ret, 0.05)
    datatable(
      data.frame(
        Metric = c("Annualized Return", "Volatility", "Sharpe", "Max Drawdown", "MAR Ratio", "VaR 95%"),
        Value = round(c(mu, vol, sharpe, dd, mar, var_95), 4)
      ),
      options = list(dom = 't', paging = FALSE),
      rownames = FALSE
    )
  })
  output$predictionPlot <- renderPlotly({
    req(results())
    df <- results()$predictions
    
    p <- ggplot(df, aes(x = predicted_return, y = fwd_return)) +
      geom_point(alpha = 0.4, color = "#d90429") +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
      labs(title = "Predicted vs Actual fwd_return",
           x = "Predicted Return", y = "Actual fwd_return") +
      theme_minimal()
    
    ggplotly(p)
  })
  
  output$rmseBox <- renderValueBox({
    req(results())
    valueBox(
      subtitle = "RMSE",
      value = round(results()$rmse, 6),
      icon = icon("chart-line"),
      color = "red"
    )
  })
  
}

# === Run App ===
shinyApp(ui = ui, server = server)