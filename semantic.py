import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
# The interesting result here is that 'monkey' and 'banana' have more similarity than 'cat' and 'banana'
# This suggests that there is more to similarity scores than on a purely character driven level, context is involved.

## WORKING WITH VECTORS EXAMPLE CODE ##
tokens = nlp('cat apple monkey banana')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
# The biggest similarities came between fruits rather than between animals which I thought was interesting. Perhaps animals are more diverse.
# I also found it interesting how monkey and banana had a similarity based upon the fact that monkeys eat bananas.
# Cat and banana and cat and apple had the same similarity scores suggesting little link between the fruit and the animal.
# I added in dog to see what happened and dog and cat had a much higher similarity than cat and monkey. Perhaps because they are common pets.

## MY OWN EXAMPLE COMPARING SPORTS ##
tokens = nlp('tennis badminton golf football cricket rugby')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))
# This was interesting as tennis and badminton had the highest similarity score so seemingly understands the sports.
# Similarly, football and rugby had the highest similarity score.

## WORKING WITH SENTENCES EXAMPLE CODE ##
sentence_to_compare = "why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


# I ran the example file and I noticed that the simpler language model 'en_core_web_sm' returns lower similarity scores then 'en_core_web_md' comparing the same sentences.
# This is because the simpler language model has a much smaller library to work from.
# The en_core_web_md model has 514k keys and 20k unique vectors (300 dimensions) whereas en_core_web_sm has none.
# Word vectors are numerical, multi-dimensional representations of words allowing the computer to understand the text better.
# With lower numerical representations of words, the similarity score won't be as high as the computer cannot determine syntactical similarity.
# The similarity comparison is calculated in a simpler way.
