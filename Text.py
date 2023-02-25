#  This code will read an input csv file and break specified text fields into words and n-grams.
#  Basic sentiment scores will also be assigned to each word and n-gram.
#  Stop words will also be identified in the word breakout.
#  Two csv files (one for words and one for n-grams) will be written.
#
#  Written by Ken Flerlage, August, 2019
#
#  Note: This code makes use of the Natural Language Toolkit (NLTK) library.
#  Be sure to download all the nltk data by using the following command: python -m nltk.downloader all 
#  ...or you can run the following commands to save time and disk space:
#     python -m nltk.downloader vader_lexicon
#     python -m nltk.downloader stopwords
#  To add more stop words, edit the language file in nltk_data\corpora\stopwords
#
#  This code uses a number of other libraries which may need to be installed before using.
#
#  --------------------------------------------------------------------------------------------------------
#  Copyright (c) 2019 Kenneth Flerlage
#  
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
#  associated documentation files (the "Software"), to deal in the Software without restriction, including 
#  without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
#  copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to 
#  the following conditions: The above copyright notice and this permission notice shall be included in 
#  all copies or substantial portions of the Software.
#  
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
#  LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN 
#  NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
#  --------------------------------------------------------------------------------------------------------

import csv
import re
import math
import os
import sys
from nltk.util import ngrams
from nltk.corpus import stopwords 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer 
import PySimpleGUI as sg

# Get user input.
stopWordLanguageList = ['arabic','azerbaijani','danish','dutch','english','finnish','french','german','greek','hungarian','indonesian','italian','kazakh','nepali','norwegian','portuguese','romanian','russian','slovene','spanish','swedish','tajik','turkish']
stopWordLanguage = 'english'

form = sg.FlexForm('Text Analysis Parameters')  # begin with a blank form

layout = [
          [sg.Text('How would you like to parse your text?')],
          [sg.Text('Full Path of Input File (Use \\\ as Separators)', size=(42, 1)), sg.InputText('C:\\\Your Folder\\\Your File.csv')],
          [sg.Text('Comma-Separated List of Fields to Analyze (No Spaces)', size=(42, 1)), sg.InputText('Field1,Field2')],
          [sg.Text('N-Gram Size (Number of Words)', size=(42, 1)), sg.InputText('6')],
          [sg.Text('Number of Sections', size=(42, 1)), sg.InputText('10')],
          [sg.Text('Langugage to Use for Stop Words', size=(42, 1)), sg.Combo(values=stopWordLanguageList,default_value='english')],
          [sg.Submit(), sg.Cancel()]
         ]

button, values = form.Layout(layout).Read()

# Continue if the user did not cancel, did not close the dialog, and entered the right type of value
if button == "Cancel" or button is None or not(values[2].isdigit()) or not(values[3].isdigit()):
    sys.exit("You either canceled/closed the dialog or entered an invalid parameter. Exiting the program.") 

# Read the input
inputFile = values[0]
wordFilePath = os.path.dirname(inputFile) + "\\"
textFields = values[1].split(',')
stopWordLanguage = values[4]
numberOfWords = int(values[2])
numberOfSections = int(values[3])

# Valid stop word language?
if not(stopWordLanguage in stopWordLanguageList):
     sys.exit("Invalid stop word language. Exiting program.")   

# Check to make sure the input file exists:
if not(os.path.exists(inputFile)):
    sys.exit("Input file does not exits. Exiting the program.")

# Delete any previously written files.
outFile = wordFilePath + "Words.csv"
if os.path.exists(outFile):
    os.remove(outFile) 

outFile = wordFilePath + "NGrams.csv"
if os.path.exists(outFile):
    os.remove(outFile) 

# Get list of stop words (will be used later)
stopwords = set(stopwords.words(stopWordLanguage)) 

stemmer = SnowballStemmer(stopWordLanguage) 

recordCounter = 0

# Open the input csv file. Loop through each record and process each field (textFields)
csv.field_size_limit(min(sys.maxsize, 2147483646))
with open(inputFile, mode='r', encoding='utf-8') as csvFile:
    csvReader = csv.DictReader(csvFile)
    lineCount = 0
    for csvRow in csvReader:
        recordID = csvRow["Record ID"]

        # Process each text field.
        for textItem in textFields:
            text = csvRow[textItem]

            recordCounter = recordCounter + 1
            
            # Text cleanup
            text = text.replace('\n', ' ') 
            text = " " + text.lower() 
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # Replace all none alphanumeric characters with spaces

            # Break into single words
            tokens = [token for token in text.split(" ") if token != ""]
            output = list(ngrams(tokens, 1))

            sectionWordCount = math.ceil(len(output)/numberOfSections)

            # Create the sentiment analyzer for some basic sentiment tests.
            nltkSentiment = SentimentIntensityAnalyzer()

            # Write single words to csv
            outFile = wordFilePath + "Words.csv"

            with open(outFile,'a', newline='', encoding='utf-8') as out:
                csvOut = csv.writer(out)

                # Write the heading
                if recordCounter == 1:
                    heading = ('Field','Record ID','Word','Stem','Stop Word','Sentiment','Word Number','Section','Section Word Number')
                    csvOut.writerow(heading)

                wordNumber = 1
                wordNumberInSection = 1
                section = 1

                # Write each word
                for row in output:
                    word = ''.join(row) #  Convert the tuple to a string

                    # Get the word's stem.
                    wordStem = stemmer.stem(word)

                    # Get the word's sentiment score.
                    score = nltkSentiment.polarity_scores(word)
                    compoundScore = score['compound']

                    if word in stopwords:
                        isStopWord = True
                    else:
                        isStopWord = False

                    row = (str(textItem),) + (str(recordID),) + row + (wordStem,) + (str(isStopWord),) + (str(compoundScore),) + (str(wordNumber),) + (str(section),) + (str(wordNumberInSection),)
                    csvOut.writerow(row)

                    # Update counter and section.
                    if wordNumberInSection % sectionWordCount == 0:
                        section = section + 1
                        wordNumberInSection = 1
                    else:
                        wordNumberInSection = wordNumberInSection + 1

                    wordNumber = wordNumber + 1

            # Parse into n-grams
            tokens = [token for token in text.split(" ") if token != ""]
            output = list(ngrams(tokens, numberOfWords))

            # Write n-grams to csv
            outFile = wordFilePath + "NGrams.csv"

            with open(outFile,'a', newline='', encoding='utf-8') as out:
                csvOut = csv.writer(out)

                # Write the heading
                if recordCounter == 1:
                    heading = ('Field','Record ID','Word1',)
                    for i in range(2, numberOfWords+1):
                        heading = heading + ('Word' + str(i),)
                    
                    heading = heading + ('Full N-Gram','N-Gram Sentiment','N-Gram Number','Section','Section N-Gram Number')
                    csvOut.writerow(heading)

                wordNumber = 1
                wordNumberInSection = 1
                section = 1

                # Write each n-gram
                fullLine = ''
                for row in output:
                    fullLine = ' '.join(row) # Build the full string with spaces
                    
                    # Get the n-gram's sentiment score.
                    score = nltkSentiment.polarity_scores(fullLine)
                    compoundScore = score['compound']

                    row = (str(textItem),) + (str(recordID),) + row + (fullLine,) + (str(compoundScore),) + (str(wordNumber),) + (str(section),) + (str(wordNumberInSection),)
                    csvOut.writerow(row)

                    # Update counter and section.
                    if wordNumberInSection % sectionWordCount == 0:
                        section = section + 1
                        wordNumberInSection = 1
                    else:
                        wordNumberInSection = wordNumberInSection + 1

                    wordNumber = wordNumber + 1

        lineCount += 1