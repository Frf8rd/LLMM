1. Prepare the data

I read a file with English words and their "dinosaur" counterparts

I clean the text: remove strange characters, lowercase, add spaces

I add special characters: \t at the beginning and \n at the end

2. Create the "dictionaries"

I make a list of all the letters used in English

I make a list of all the letters used in dinosaur

I give each letter a coded number

3. Convert the words to numbers

I convert each word into a sequence of numbers

I pad with zeros so that all sequences have the same length

4. Build the artificial "brain"

I have two parts: one that understands English, the other that generates dinosaur

The English part reads the text in both directions (beginning->end and end->beginning)

The information passes to the dinosaur part which starts generating the translation

The model "guesses" letter by letter what should follow

5. Train the model

Train it using the examples in the file

Compare the results with the correct translations

I gradually improve the performance

I stop training when it stops improving

6. Save the model

I keep the trained "brain" so I can use it later

##PART-2
1. Load the trained model

The program no longer trains a new model, but uses a previously trained one

It's like using a pre-written dictionary

2. Prepare the text for translation

Clean the input text: remove special characters, make it lowercase

Convert each letter into numbers (encoding) so that it can be understood by the computer

3. Use two artificial "brains"

The first brain (encoder): reads and understands the text in English

The second brain (decoder): generates the translation in the language of the dinosaurs

4. Produce the translation letter by letter letter

Starts with a special signal \t (like "start speaking dinosaur")

For each letter, "guesses" what the next correct letter in dinosaur would be

Continues until it reaches the signal \n (like "stop speaking")

Uses "temperature" to control how creative or accurate the translation is

5. Save the results

Writes all translations to a CSV file

Organizes the results in a special folder "translations"

##For run
1.Create a virtual environment (venv)
python -m venv venv

2.Activate
venv\Scripts\activate

3.Install the required libraries.
pip install -r requirements.txt

4.Run file frist
5.Run file main