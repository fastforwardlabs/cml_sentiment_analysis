## Part 4: Application

# This script explains how to create and deploy Applications in CML.
# This feature allows data scientists to **get ML solutions in front of stakeholders quickly**,
# including business users who need results fast.
# This may be good for sharing a **highly customized dashboard**, a **monitoring tool**, or a **product mockup**.

# CML is agnostic regarding frameworks.
# [Flask](https://flask.palletsprojects.com/en/1.1.x/),
# [Dash](https://plotly.com/dash/) apps will both work.
# R users will find it easy to deploy [Shiny](https://shiny.rstudio.com) apps.

# If you haven't yet, run through the initialization steps in the README file and run the R Code path. Do that
# now

# This file is provides a sample Shiny app script, ready for deployment,
# which displays a sentiment prediction for a new sentence using the Model API deployed in 
# Part 3

## Deploying the Application
#
# > ### _**Note:**_ **This next step is important**
# > For both models you need to get the **Access Key** , go to **Model > Settings** and make a 
# > note (i.e. copy) the "Access Key". It will look something like this (ie. 
# > mukd9sit7tacnfq2phhn3whc4unq1f38)
# >
# > From the Project level click on "Open Workbench" (note you don't actually have to Launch a 
# > session) in order to edit a file. Select the `R Code\4_shiny_app.R` file and paste the Access 
# > Keys in to the `fetch_result` function. The first Access Key is for the R Model and the second
# > is for the Python model. 
# > 
# > ```
# > fetch_result <- function (sentence, model) {
# >   if (model == "simp") {
# >     accessKey <-  "mfd0yk8o4tfi13uua8hc9gzqxej0jc2s"
# >   }
# >   else {
# >     accessKey <-  "m7zzyhlbtr3ovq3tvaa2myowglhzpf3f"
# >   }
# > ```
# > Save the file (if it has not auto saved already) and go back to the Project.

# From there Go to the **Applications** section and select "New Application" with the following:
# * **Name**:Sentiment Predictor
# * **Subdomain**: sentibot _(note: this needs to be unique, so if you've done this before, 
# pick a more random subdomain name)_
# * **Script**: R Code/4_shiny_app.R
# * **Kernel**: R
# * **Engine Profile**: 1vCPU / 2 GiB Memory

### Using the Application
# After the Application deploys, click on the blue-arrow next to the name to launch the 
# applicatio. This application is self explanitory, type in a sentence and choose which model
# to send it to to get a sentiment prediction back.

library(shiny)
library(dplyr)
library(httr)

fetch_result <- function (sentence, model) {
  if (model == "simp") {
    accessKey <-  "mfd0yk8o4tfi13uua8hc9gzqxej0jc2s"
  }
  else {
    accessKey <-  "m7zzyhlbtr3ovq3tvaa2myowglhzpf3f"
  }
  result <- POST(
    paste("https://modelservice.", Sys.getenv("CDSW_DOMAIN"), "/model", sep=""),
    body = paste('{"accessKey":"',accessKey,'","request":{"sentence":"',sentence,'"}} ',sep = ""),
    add_headers("Content-Type" = "application/json")
  )
  
  model_response <- fromJSON(rawToChar(result$content))#$response
  return_ouput <- paste("The model is", model_response$response["confidence"],"% confident that is", model_response$response["sentiment"])
  return(return_ouput)
}


app <- shinyApp(ui <- fluidPage(
  titlePanel("Sentiment Analysis Model Application"),
  
  sidebarLayout(
    sidebarPanel(
      textAreaInput( 
        "caption", "Test Sentence", "I have had a bad day"
      ),
      radioButtons(
        "model", "Choose model:", c("R" = "simp", "Python" = "dl")
      ),
      submitButton("Get Sentiment", icon("arrow-right"))
    ),
    
    mainPanel(
      markdown(
        "
        #### Model Result Output
        The _Test Sentence_ will be sent to the selected model and the response will be displayed below
        "
      ),
      
      verbatimTextOutput("value")
    )
  )
),

server <- function(input, output) {
  output$value <- renderText({
    fetch_result(input$caption, input$model)
  })
})

runApp(app, port = as.numeric(Sys.getenv("CDSW_READONLY_PORT")), host = "127.0.0.1", launch.browser = "FALSE")
