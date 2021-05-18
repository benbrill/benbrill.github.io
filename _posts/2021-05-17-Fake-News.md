---
layout: post
title: Classifying Fake News
image: benbrill.github.io\images\ucla-math.png
---

# Classifying Fake News using Neural Networks 
COVID-19 is not the only pandemic we are facing. Though it is far less deadly, our country and our world has been experiencing an epidemic of fake news for the past few years. The rising of use of social media in political campaigns has slowly but surely introduced more and more false information into our news ecosystem, as to convince the masses to form one opinion over another. It has already substantially affected a few elections, and could very well affect more in the near future.

Rooting out this problem is not that difficult; just check the facts of the story to determine if it is real or not. However, in practice, that can be impractical to readers who these stories are targeted at, who often simply look at the headline to get the gist of the story.

What if there was some way to classify these stories as real or fake for readers before they view it, and to do this without verifying each and every fact in the story but simply to look at patterns in the headlines and text? Enter **neural networks**.

## Setting up the data
First, let's import the whole boatload of libraries that are necessary to do this. 


```python
# basic libraries
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk import download
import numpy as np
import string
import re

# libraries to build model
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras

# specialized layers to format data appropriate for the model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
```

We will also must download a list of stopwords (a, the, but, and, etc.) that we will use later


```python
download('stopwords')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True



Let's load in our data from the url below


```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
data = pd.read_csv(train_url)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22444</th>
      <td>10709</td>
      <td>ALARMING: NSA Refuses to Release Clinton-Lynch...</td>
      <td>If Clinton and Lynch just talked about grandki...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22445</th>
      <td>8731</td>
      <td>Can Pence's vow not to sling mud survive a Tru...</td>
      <td>() - In 1990, during a close and bitter congre...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22446</th>
      <td>4733</td>
      <td>Watch Trump Campaign Try To Spin Their Way Ou...</td>
      <td>A new ad by the Hillary Clinton SuperPac Prior...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22447</th>
      <td>3993</td>
      <td>Trump celebrates first 100 days as president, ...</td>
      <td>HARRISBURG, Pa.U.S. President Donald Trump hit...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22448</th>
      <td>12896</td>
      <td>TRUMP SUPPORTERS REACT TO DEBATE: “Clinton New...</td>
      <td>MELBOURNE, FL is a town with a population of 7...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22449 rows × 4 columns</p>
</div>



We see we have a data frame with each row representing one story. We have a `title` column which is the headline for each story, the text of the story in the `text` column, and whether the story was fake or not in the `fake` column, with a `1` denoting that the story is fake.

### Preprocessing Text
In order to get the best result out of our model, we only want to give it words to analyze that are import. "Stopwords," like "but," "and," or "I," really don't have any particular meaning nor give any weight to determining whether a story is real or not. So we will get rid of each of these words in the `title` and `text` columns.

In addition, we are going to convert our dataframe columns into a `tensorflow` Dataset, which is essentially an easier way for our neural networks to read our data.


```python
def make_dataset(df):
  stop = stopwords.words('english')
  df = df.copy()
  df['title'] = df['title'].apply(lambda x: ' '.join(item for item in x.split() if item not in stop))
  df['text'] = df['text'].apply(lambda x: ' '.join(item for item in x.split() if item not in stop))

  data = tf.data.Dataset.from_tensor_slices((
      # predictor data
      {
          "title": df[["title"]],
          "text": df[["text"]]
      },
      {
          "fake": df[["fake"]]
      }
  ))

  return data

```


```python
tfData = make_dataset(data)
```

### Train-test split
Now that we have our dataset set up, we are going to split it into a data set for training, a data set for validation, as well as a data set for testing and evaluation. Since our data is formated in a special `tensorflow` Dataset, we will use the `take()` method.


```python
tfData = tfData.shuffle(buffer_size = len(tfData))

train_size = int(0.7*len(tfData))
val_size   = int(0.1*len(tfData))

train = tfData.take(train_size).batch(20)
val   = tfData.skip(train_size).take(val_size).batch(20)
test  = tfData.skip(train_size + val_size).batch(20)

len(train), len(val), len(test)
```




    (786, 113, 225)



### Creating Vectorized text
Finally, we are going to standardize each of our columns with text by converting all words to lower case and removing any punctuation.

Once that is done, we will vectorize our text. What this means is that we are going to seperate an entire string of words into seperate vector components, with a numerical representation of each word. Since computers only understand numbers, this is a very important step


```python
size_vocabulary = 2000
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 


```

We can see what this vectorization looks like below once we adapt each layer to a specific purpose

# Title Model

We are not going to create a model that can predict whether a story is real or fake based off it's headline. First, we must tell our vecorization layer we want it to vectorize our `title` column


```python
vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
```

We must now specify the input for our model. In this case it will be the `title` of a story. We can use the special `keras.Input` fclass to specify it's name, dimensions, and its data type.


```python
titleInput = keras.Input(
    shape = (1,), 
    name = "title",
    dtype = "string"
)
```

Now that we have an input, we can create a neural network. Neural Networks are comprised of layers of various types, each with a different function. In this model, we will be using the `vectorize_layer`, which vectorizes our text, and `Embedding` layer, which create a vector based off of the vectorization layer who's magnitude supposedly corresponds to the word's significane in the context of fake news. The `10` in the parameters in that layer signifies we want to have vectors in 10 dimensional space, which is sort of an arbitray number but can yield good results.

Then, we will use `Dropout` layers to drop a certain amount of data if overfitting is occuring, and `GlobalAveragePooling1D` layers to convert our 2D vectors into something more digestable.


```python
titleModel = vectorize_layer(titleInput)
titleModel = layers.Embedding(size_vocabulary, 10, name = "embedding")(titleModel)
titleModel = layers.Dropout(0.2)(titleModel)
titleModel = layers.GlobalAveragePooling1D()(titleModel)
titleModel = layers.Dropout(0.2)(titleModel)
titleModel = layers.Dense(32, activation='relu')(titleModel)
```

Finally, we will create an output layer with a size of 2, which is the number of outcomes we have, real or fake news. 


```python
output = layers.Dense(2, name = "fake")(titleModel)
```

Now, we can create the model by specifying its `keras.Inputs` as well as its output


```python
model = keras.Model(
    inputs = [titleInput],
    outputs = output
)
```


```python
model.summary()
```

    Model: "model_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    title (InputLayer)           [(None, 1)]               0         
    _________________________________________________________________
    text_vectorization_1 (TextVe (None, 500)               0         
    _________________________________________________________________
    embedding (Embedding)        (None, 500, 10)           20000     
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 500, 10)           0         
    _________________________________________________________________
    global_average_pooling1d_5 ( (None, 10)                0         
    _________________________________________________________________
    dropout_11 (Dropout)         (None, 10)                0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 32)                352       
    _________________________________________________________________
    fake (Dense)                 (None, 2)                 66        
    =================================================================
    Total params: 20,418
    Trainable params: 20,418
    Non-trainable params: 0
    _________________________________________________________________
    

Now that we have a pipeline set up, we can compile our model and fit it according to our training data.


```python
model.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
history = model.fit(train, 
                    validation_data=val,
                    epochs = 10, 
                    verbose = True)
```

    Epoch 1/10
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning:
    
    Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
    
    

    786/786 [==============================] - 4s 5ms/step - loss: 0.6807 - accuracy: 0.5516 - val_loss: 0.3782 - val_accuracy: 0.9104
    Epoch 2/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.2685 - accuracy: 0.9328 - val_loss: 0.1273 - val_accuracy: 0.9559
    Epoch 3/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.1170 - accuracy: 0.9612 - val_loss: 0.0986 - val_accuracy: 0.9635
    Epoch 4/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0889 - accuracy: 0.9689 - val_loss: 0.0577 - val_accuracy: 0.9768
    Epoch 5/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0669 - accuracy: 0.9755 - val_loss: 0.0542 - val_accuracy: 0.9835
    Epoch 6/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0627 - accuracy: 0.9787 - val_loss: 0.0537 - val_accuracy: 0.9782
    Epoch 7/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0562 - accuracy: 0.9801 - val_loss: 0.0525 - val_accuracy: 0.9844
    Epoch 8/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0591 - accuracy: 0.9799 - val_loss: 0.0456 - val_accuracy: 0.9835
    Epoch 9/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0510 - accuracy: 0.9821 - val_loss: 0.0371 - val_accuracy: 0.9884
    Epoch 10/10
    786/786 [==============================] - 4s 5ms/step - loss: 0.0454 - accuracy: 0.9835 - val_loss: 0.0422 - val_accuracy: 0.9862
    

Our model does pretty well on both training and validation data! Let's see how a similar model will perform on the `text` of a story

# Text Model


```python
vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```


```python
textModel = vectorize_layer(textInput)
textModel = layers.Embedding(size_vocabulary, 10, name = "embedding")(textModel)
textModel = layers.Dropout(0.2)(textModel)
textModel = layers.GlobalAveragePooling1D()(textModel)
textModel = layers.Dropout(0.2)(textModel)
textModel = layers.Dense(32, activation='relu')(textModel)
```


```python
output = layers.Dense(2, name = "fake")(textModel)
model = keras.Model(
    inputs = [textInput],
    outputs = output
)
```


```python
model.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
textHistory = model.fit(train, 
                    validation_data=val,
                    epochs = 10, 
                    verbose = True)
```

    Epoch 1/10
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning:
    
    Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
    
    

    786/786 [==============================] - 6s 7ms/step - loss: 0.5709 - accuracy: 0.6932 - val_loss: 0.1613 - val_accuracy: 0.9523
    Epoch 2/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.1611 - accuracy: 0.9542 - val_loss: 0.1045 - val_accuracy: 0.9746
    Epoch 3/10
    786/786 [==============================] - 5s 6ms/step - loss: 0.1052 - accuracy: 0.9704 - val_loss: 0.1087 - val_accuracy: 0.9697
    Epoch 4/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0957 - accuracy: 0.9724 - val_loss: 0.0916 - val_accuracy: 0.9742
    Epoch 5/10
    786/786 [==============================] - 5s 6ms/step - loss: 0.0812 - accuracy: 0.9751 - val_loss: 0.0725 - val_accuracy: 0.9746
    Epoch 6/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0706 - accuracy: 0.9785 - val_loss: 0.0583 - val_accuracy: 0.9831
    Epoch 7/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0603 - accuracy: 0.9831 - val_loss: 0.0630 - val_accuracy: 0.9799
    Epoch 8/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0634 - accuracy: 0.9804 - val_loss: 0.0518 - val_accuracy: 0.9844
    Epoch 9/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0518 - accuracy: 0.9843 - val_loss: 0.0454 - val_accuracy: 0.9853
    Epoch 10/10
    786/786 [==============================] - 5s 7ms/step - loss: 0.0470 - accuracy: 0.9856 - val_loss: 0.0691 - val_accuracy: 0.9639
    

# Title and Text Model


```python
vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
```


```python
titleInput = keras.Input(
    shape = (1,), 
    name = "title",
    dtype = "string"
)

textInput = keras.Input(
    shape = (1,), 
    name = "text",
    dtype = "string"
)
```


```python
main = layers.concatenate([titleModel, textModel], axis = 1)
```


```python
main = layers.Dense(32, activation='relu')(main)
output = layers.Dense(2, name = "fake")(main)
```


```python
model = keras.Model(
    inputs = [titleInput, textInput],
    outputs = output
)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-32-006de513692e> in <module>()
          1 model = keras.Model(
          2     inputs = [titleInput, textInput],
    ----> 3     outputs = output
          4 )
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/base.py in _method_wrapper(self, *args, **kwargs)
        515     self._self_setattr_tracking = False  # pylint: disable=protected-access
        516     try:
    --> 517       result = method(self, *args, **kwargs)
        518     finally:
        519       self._self_setattr_tracking = previous_value  # pylint: disable=protected-access
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py in __init__(self, inputs, outputs, name, trainable, **kwargs)
        118     generic_utils.validate_kwargs(kwargs, {})
        119     super(Functional, self).__init__(name=name, trainable=trainable)
    --> 120     self._init_graph_network(inputs, outputs)
        121 
        122   @trackable.no_automatic_dependency_tracking
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/training/tracking/base.py in _method_wrapper(self, *args, **kwargs)
        515     self._self_setattr_tracking = False  # pylint: disable=protected-access
        516     try:
    --> 517       result = method(self, *args, **kwargs)
        518     finally:
        519       self._self_setattr_tracking = previous_value  # pylint: disable=protected-access
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py in _init_graph_network(self, inputs, outputs)
        202     # Keep track of the network's nodes and layers.
        203     nodes, nodes_by_depth, layers, _ = _map_graph_network(
    --> 204         self.inputs, self.outputs)
        205     self._network_nodes = nodes
        206     self._nodes_by_depth = nodes_by_depth
    

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py in _map_graph_network(inputs, outputs)
        988                              'The following previous layers '
        989                              'were accessed without issue: ' +
    --> 990                              str(layers_with_complete_input))
        991         for x in nest.flatten(node.outputs):
        992           computable_tensors.add(id(x))
    

    ValueError: Graph disconnected: cannot obtain value for tensor KerasTensor(type_spec=TensorSpec(shape=(None, 1), dtype=tf.string, name='text'), name='text', description="created by layer 'text'") at layer "text_vectorization". The following previous layers were accessed without issue: []



```python
model.summary()
```


```python
model.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
)
```


```python
history = model.fit(train, 
                    validation_data=val,
                    epochs = 10, 
                    verbose = True)
```

# Validation


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
testingData = pd.read_csv(test_url)
testingData
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>420</td>
      <td>CNN And MSNBC Destroy Trump, Black Out His Fa...</td>
      <td>Donald Trump practically does something to cri...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14902</td>
      <td>Exclusive: Kremlin tells companies to deliver ...</td>
      <td>The Kremlin wants good news.  The Russian lead...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>322</td>
      <td>Golden State Warriors Coach Just WRECKED Trum...</td>
      <td>On Saturday, the man we re forced to call  Pre...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16108</td>
      <td>Putin opens monument to Stalin's victims, diss...</td>
      <td>President Vladimir Putin inaugurated a monumen...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10304</td>
      <td>BREAKING: DNC HACKER FIRED For Bank Fraud…Blam...</td>
      <td>Apparently breaking the law and scamming the g...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>22444</th>
      <td>20058</td>
      <td>U.S. will stand be steadfast ally to Britain a...</td>
      <td>The United States will stand by Britain as it ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22445</th>
      <td>21104</td>
      <td>Trump rebukes South Korea after North Korean b...</td>
      <td>U.S. President Donald Trump admonished South K...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22446</th>
      <td>2842</td>
      <td>New rule requires U.S. banks to allow consumer...</td>
      <td>U.S. banks and credit card companies could be ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22447</th>
      <td>22298</td>
      <td>US Middle Class Still Suffering from Rockefell...</td>
      <td>Dick Eastman The Truth HoundWhen Henry Kissin...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22448</th>
      <td>333</td>
      <td>Scaramucci TV Appearance Goes Off The Rails A...</td>
      <td>The most infamous characters from Donald Trump...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>22449 rows × 4 columns</p>
</div>




```python
testingData = make_dataset(testingData)
```


```python
model.evaluate(testingData)
```

    /usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/engine/functional.py:595: UserWarning: Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
      [n for n in tensors.keys() if n not in ref_input_names])
    

    22449/22449 [==============================] - 52s 2ms/step - loss: 2.4460 - accuracy: 0.5038
    




    [2.458550214767456, 0.5039422512054443]




```python
weights = model.get_layer('embedding').get_weights()[0] # get the weights from the embedding layer
vocab = vectorize_layer.get_vocabulary()                # get the vocabulary from our data prep for later

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
weights = pca.fit_transform(weights)

embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```


```python
import plotly.express as px 
import numpy as np

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="8c451939-f6de-4f63-8af9-ade796be4434" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("8c451939-f6de-4f63-8af9-ade796be4434")) {
                    Plotly.newPlot(
                        '8c451939-f6de-4f63-8af9-ade796be4434',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "<b>%{hovertext}</b><br><br>x0=%{x}<br>x1=%{y}<br>size=%{marker.size}", "hovertext": ["", "[UNK]", "trump", "to", "video", "the", "us", "for", "of", "in", "says", "on", "a", "and", "is", "obama", "with", "watch", "house", "hillary", "new", "his", "about", "after", "clinton", "trump\u2019s", "white", "president", "just", "at", "bill", "by", "from", "state", "russia", "north", "over", "who", "he", "out", "court", "this", "republican", "it", "her", "senate", "donald", "will", "are", "as", "not", "korea", "vote", "election", "breaking", "him", "calls", "media", "black", "that", "news", "be", "republicans", "how", "\u2013", "police", "tax", "why", "was", "gop", "trumps", "you", "muslim", "china", "campaign", "may", "deal", "obama\u2019s", "has", "up", "iran", "democrats", "one", "down", "gets", "back", "un", "fbi", "what", "party", "former", "tweets", "attack", "people", "eu", "syria", "russian", "congress", "talks", "speech", "top", "have", "first", "against", "senator", "pm", "government", "leader", "fox", "security", "cnn", "chief", "judge", "war", "america", "ban", "they", "plan", "it\u2019s", "would", "south", "no", "minister", "law", "democrat", "twitter", "makes", "during", "their", "sanders", "make", "when", "say", "man", "tells", "report", "factbox", "military", "liberal", "official", "more", "gun", "get", "brexit", "two", "take", "she", "could", "hillary\u2019s", "border", "shows", "sanctions", "supreme", "presidential", "probe", "million", "host", "cruz", "if", "tweet", "into", "goes", "racist", "fight", "nuclear", "wants", "healthcare", "americans", "foreign", "all", "like", "woman", "meet", "governor", "american", "them", "political", "off", "obamacare", "being", "putin", "go", "time", "day", "an", "women", "want", "support", "rally", "poll", "illegal", "urges", "syrian", "supporters", "show", "wow", "warns", "national", "debate", "response", "islamic", "budget", "bernie", "attacks", "refugees", "our", "help", "lawmakers", "he\u2019s", "call", "your", "race", "policy", "german", "crisis", "gives", "administration", "ryan", "panel", "old", "group", "claims", "so", "uk", "turkey", "conservative", "candidate", "antitrump", "visit", "see", "saudi", "opposition", "leaders", "don\u2019t", "win", "states", "rights", "because", "asks", "world", "ted", "most", "won\u2019t", "while", "stop", "next", "meeting", "killed", "travel", "trade", "must", "students", "room", "mexico", "death", "comey", "climate", "school", "right", "going", "general", "cops", "press", "officials", "arrested", "we", "reporter", "details", "air", "sources", "hilarious", "defense", "immigration", "fake", "can", "before", "texas", "department", "big", "way", "ties", "pick", "huge", "email", "case", "takes", "than", "plans", "violence", "did", "secretary", "interview", "got", "years", "supporter", "should", "move", "major", "iraq", "federal", "end", "aid", "year", "head", "democratic", "caught", "can\u2019t", "tillerson", "seeks", "really", "protesters", "left", "justice", "emails", "use", "open", "money", "made", "city", "shocking", "kill", "here\u2019s", "health", "exclusive", "director", "paul", "voters", "reason", "mayor", "i", "free", "conservatives", "work", "rules", "now", "lives", "live", "john", "but", "business", "washington", "wall", "secret", "lawmaker", "dead", "ahead", "threatens", "presidency", "need", "leftist", "boom", "sexual", "lie", "bomb", "boiler", "tv", "myanmar", "merkel", "isis", "give", "george", "college", "change", "run", "reform", "terrorists", "job", "do", "cut", "control", "truth", "team", "still", "push", "list", "last", "jerusalem", "investigation", "family", "attorney", "voter", "times", "speaker", "only", "funding", "britain", "told", "said", "ready", "puerto", "office", "fire", "even", "coalition", "amid", "own", "know", "key", "high", "billion", "terrorist", "release", "protest", "nominee", "muslims", "lawyer", "home", "florida", "ever", "chair", "order", "independence", "image", "votes", "shooting", "rohingya", "post", "pence", "never", "latest", "japan", "germany", "face", "british", "bid", "korean", "keep", "dnc", "discuss", "called", "trying", "ep", "busted", "army", "2016", "social", "slams", "force", "behind", "week", "threat", "statement", "ruling", "great", "flynn", "flag", "decision", "charges", "another", "rico", "protests", "power", "needs", "msnbc", "catalan", "best", "adviser", "york", "vows", "story", "refugee", "peace", "legal", "france", "blasts", "seek", "sean", "mccain", "march", "lol", "full", "doesn\u2019t", "backs", "again", "violent", "source", "son", "senators", "scandal", "radical", "or", "kremlin", "israel", "intelligence", "hate", "found", "cia", "been", "message", "macron", "himself", "had", "forces", "fired", "claim", "asked", "agency", "admits", "\u201cthe", "young", "used", "parliament", "nyc", "iraqi", "good", "destroys", "defends", "believe", "audio", "arrest", "wife", "terror", "rule", "making", "lead", "jobs", "committee", "clinton\u2019s", "california", "bombshell", "away", "action", "3", "10", "think", "special", "set", "lies", "kills", "inauguration", "cuba", "chicago", "were", "water", "sex", "real", "london", "internet", "dem", "congressman", "benghazi", "aide", "working", "too", "reveals", "public", "nfl", "matter", "hurricane", "final", "chinas", "warren", "voting", "turkish", "tries", "saying", "review", "cuts", "calling", "bush", "ad", "using", "under", "sign", "paris", "much", "hollywood", "hold", "hilariously", "didn\u2019t", "debt", "bad", "\u201ci", "yet", "tell", "shut", "sessions", "senior", "reports", "rep", "questions", "pay", "near", "michelle", "least", "kurdish", "kids", "hit", "french", "does", "despite", "corruption", "children", "act", "venezuela", "trip", "talk", "put", "king", "ivanka", "hard", "fraud", "blames", "barack", "wins", "three", "threats", "start", "repeal", "question", "possible", "mike", "leave", "hell", "fed", "envoy", "denies", "dc", "ben", "awesome", "\u201cwe", "xi", "these", "rubio", "return", "proves", "pressure", "orders", "ny", "lying", "look", "little", "likely", "kellyanne", "jr", "hits", "five", "female", "exposes", "executive", "epa", "destroy", "days", "cop", "1", "spokesman", "shot", "she\u2019s", "seen", "role", "pentagon", "mcconnell", "jail", "drops", "break", "approves", "announcement", "angry", "ambassador", "whoa", "victims", "thing", "taking", "summit", "releases", "refuses", "program", "moore", "let", "kelly", "girl", "chris", "accuses", "abortion", "they\u2019re", "stunning", "service", "pope", "planned", "passes", "missile", "laws", "issue", "global", "fighting", "disgusting", "demands", "convention", "comments", "brutal", "block", "battle", "assault", "announces", "agree", "yemen", "without", "very", "united", "street", "star", "second", "russias", "photo", "men", "immigrants", "images", "hack", "groups", "gave", "facebook", "epic", "energy", "country", "conway", "conference", "citizens", "child", "ceo", "car", "brilliant", "biden", "al", "workers", "wikileaks", "uses", "transgender", "spending", "sarah", "released", "rant", "nancy", "members", "killing", "getting", "future", "forced", "evidence", "every", "erdogan", "around", "4", "ukraine", "trial", "test", "strike", "spain", "schools", "rips", "rejects", "protect", "pelosi", "oil", "offers", "middle", "melania", "liberals", "joe", "friday", "food", "flashback", "doj", "defend", "crooked", "arms", "agenda", "yr", "viral", "terrorism", "strikes", "stand", "running", "rape", "private", "offer", "nato", "migrants", "life", "issues", "illegals", "history", "foundation", "finally", "faces", "elections", "afghan", "5", "weapons", "victory", "university", "troops", "spicer", "sees", "oregon", "names", "millions", "michael", "loses", "letter", "lebanon", "join", "fans", "dangerous", "congressional", "confirms", "challenge", "chairman", "catalonia", "carson", "canada", "boy", "already", "accused", "2018", "warning", "suspected", "sunday", "step", "staff", "sht", "responds", "puts", "points", "member", "israeli", "elizabeth", "bank", "alien", "wrong", "tried", "system", "picks", "philippines", "oops", "militants", "mexican", "labor", "insane", "gay", "deputy", "dems", "criminal", "come", "coal", "close", "civil", "cabinet", "baltimore", "anchor", "activist", "vp", "turn", "thousands", "steps", "speaks", "since", "number", "name", "mom", "leaves", "irish", "intel", "four", "find", "finance", "fck", "eus", "europe", "embassy", "crowd", "breaks", "bathroom", "australia", "arabia", "absolutely", "911", "worst", "virginia", "town", "taiwan", "stay", "sheriff", "same", "prison", "perfect", "parties", "owner", "other", "night", "mueller", "moscow", "meltdown", "megyn", "mattis", "james", "hopes", "fund", "effort", "east", "crazy", "church", "bring", "allies", "zimbabwe", "west", "transition", "thinks", "steve", "something", "sea", "rich", "ria", "revealed", "referendum", "percent", "northern", "michigan", "line", "irma", "ireland", "hacking", "guest", "girls", "firm", "experts", "dad", "cyber", "cnn\u2019s", "christmas", "chinese", "bundy", "aliens", "alabama", "activists", "access", "6", "20", "15", "update", "target", "student", "stage", "signs", "resign", "raises", "part", "palestinian", "navy", "murder", "mocks", "meets", "massive", "journalist", "immigrant", "hearing", "heads", "hannity", "germanys", "frances", "financial", "far", "exposed", "embarrassing", "economy", "economic", "detroit", "destroyed", "declares", "company", "coming", "choice", "brutally", "asia", "antifa", "words", "vietnam", "urge", "union", "turkeys", "syrias", "soros", "reutersipsos", "racism", "progress", "politics", "picture", "nafta", "mark", "launches", "guy", "friend", "eyes", "explains", "efforts", "daughter", "criticism", "cities", "boost", "better", "bans", "bangladesh", "anthem", "well", "vs", "uks", "totally", "sue", "recount", "record", "prosecutor", "pass", "ohio", "moment", "manager", "lawsuit", "international", "illinois", "human", "hot", "henningsen", "guns", "firing", "due", "drug", "desperate", "demand", "conspiracy", "confederate", "companies", "comment", "christie", "charged", "charge", "changes", "care", "blame", "attacked", "any", "\u2018the", "waters", "try", "telling", "spy", "speak", "shuts", "save", "quits", "potential", "my", "moves", "migrant", "mass", "lose", "johnson", "its", "information", "industry", "harassment", "funds", "fear", "everyone", "ends", "employees", "elected", "diplomatic", "christian", "candidates", "boycott", "allow", "agencies", "african", "actor", "where", "voted", "unhinged", "then", "tensions", "taxes", "super", "send", "residents", "remarks", "red", "qatar", "parenthood", "o\u2019reilly", "obamas", "mugabe", "maxine", "literally", "kurds", "jeanine", "humiliates", "highlights", "hammers", "green", "golf", "giving", "finds", "feds", "event", "doing", "council", "check", "ca", "britains", "base", "actually", "worse", "videos", "veteran", "total", "today", "socialist", "screenshots", "roy", "resigns", "request", "reporters", "reach", "raise", "praises", "powerful", "play", "palin", "nbc", "might", "marriage", "market", "love", "leadership", "journalists", "happened", "hacked", "gov", "game", "front", "flint", "father", "enough", "deals", "corporate", "community", "comes", "carolina", "carlson", "capital", "between", "ask", "allegations", "afghanistan", "address", "50", "2", "17", "victim", "tucker", "train", "took", "testimony", "term", "some", "sick", "shooter", "sets", "quit", "proposes", "promises", "prince", "place", "patrick", "parents", "outside", "opens", "news\u2019", "netanyahu", "lost", "looks", "long", "koreas", "held", "hand", "furious", "fires", "done", "dept", "dallas", "crime", "committed", "class", "australian", "asking", "answer", "zimbabwes", "who\u2019s", "wearing", "veterans", "unity", "targets", "talking", "sent", "russians", "reuters", "protester", "proof", "players", "pastor", "nomination", "nations", "movie", "less", "kkk", "iowa", "hundreds", "hope", "guilty", "gas", "freedom", "false", "electoral", "egypt", "education", "draws", "dispute", "dialogue", "detained", "data", "countries", "counsel", "chuck", "borders", "book", "blocks", "become", "attempt", "arrests", "armed", "appeals", "8", "winning", "visa", "tuesday", "thursday", "thug", "things", "taxpayer", "surprise", "supporting", "suicide", "seth", "sends", "sanctuary", "religious", "region", "professor", "polls", "pact", "officer", "murdered", "mother", "mays", "libya", "leaks", "kim", "kenya", "jailed", "iranian", "idea", "homeland", "gowdy", "forward", "fix", "failed", "explain", "endorses", "eastern", "early", "donations", "december", "cooperation", "concerns", "communist", "central", "blacklivesmatter", "bannon", "baby", "assad", "america\u2019s", "airport", "agents", "across", "30", "13", "12", "you\u2019re", "worker", "wanted", "trey", "thugs", "threatened", "third", "testify", "taxpayers", "stupid", "strong", "streets", "spanish", "soldiers", "risk", "results", "reelection", "rebels", "reality", "prime", "philippine", "person", "perfectly", "pakistan", "outrageous", "nuts", "nra", "mosque", "months", "missing", "many", "manafort", "loss", "lied", "leading", "joy", "jimmy", "inside", "hosts", "hariri", "guess", "ground", "gold", "facts", "documents", "disturbing", "disaster", "continues", "clintons", "citizen", "board", "blow", "billionaire", "banks", "amazing", "abe", "100", "\u201cyou", "\u201cif", "wage", "w", "throws", "threaten", "suspect", "straight", "storm", "small", "shoot", "seven", "sentence", "sec", "romney", "rnc", "rightwing", "relations", "reaction", "popular", "others", "options", "meddling", "low", "longer", "living", "legislation", "lady", "kushner", "jeff", "japans", "injured", "independent", "hypocrisy", "husband", "hurt", "hotel", "hands", "fears", "farright", "expected", "ethics", "endorsement", "dollars", "defending", "criticizes", "crimes", "crackdown", "controversial", "concerned", "cannot", "breitbart", "avoid", "anyone", "ally", "africas", "abuse", "abc", "18", "11", "won", "weighs", "waiting", "vice", "van", "usa", "turns", "treatment", "tough", "there", "teen", "suspends", "suggests", "starts", "situation", "safe", "riots", "returns", "replace", "relief", "regime", "read", "radio", "putting", "propaganda", "presidents", "orlando", "nightmare", "mic", "met", "lynch", "libyan", "leaving", "leaked", "leads", "launch", "kerry", "info", "harvey", "half", "focus", "families", "european", "drop", "die", "delivers", "cover", "continue", "commander", "clash", "charity", "center", "campus", "brother", "brings", "brazil", "bills", "beijing", "banned", "attempts", "attacking", "aren\u2019t", "appears", "agreement", "actions", "\u201cracist\u201d", "wisconsin", "whining", "weeks", "warned", "trump\u201d", "thanks", "swedish", "study", "strategy", "station", "spying", "serious", "scott", "robert", "respond", "reasons", "ratings", "process", "posts", "planning", "overhaul", "opposes", "nation", "monday", "militia", "mi", "marco", "maher", "looms", "late", "isn\u2019t", "important", "holds", "helping", "hateful", "gonna", "deep", "decide", "consider", "condemns", "charlottesville", "chaos", "cash", "broke", "blacks", "asylum", "apart", "aides", "advisor", "ads", "7", "2017", "\u201che", "\u2014", "zika", "you\u2019ll", "women\u2019s", "whines", "watchdog", "visits", "unreal", "treasury", "teacher", "taken", "sweden", "supports", "speaking", "soon", "somali", "snl", "schumer", "san", "sales", "rise", "responsible", "resignation", "reportedly", "remove", "promote", "problems", "priceless", "policies", "phony", "phone", "path", "past", "nothing", "nearly", "mnuchin", "ministry", "majority", "losing", "lets", "lawyers", "kimmel", "jones", "joke", "joins", "jeb", "huckabee", "hispanic", "having", "halt", "georgia", "files", "fail", "entire", "easy", "dossier", "diplomats", "dinner", "dies", "delay", "david", "conflict", "complete", "commerce", "collusion", "cites", "cancels", "bus", "burn", "brussels", "body", "blast", "bizarre", "biggest", "attend", "appeal", "amnesty", "america\u201d", "alert", "agent", "accidentally", "accept", "\u201cit\u2019s", "word", "wire", "view", "vegas", "trumprussia", "thought", "stealing", "stance", "six", "showing", "sharia", "ridiculous", "relationship", "reject", "reid", "regional", "reforms", "records", "pushes", "pulls", "poland", "point", "plane", "pennsylvania", "nyt", "numbers", "nails", "mock", "missiles", "militant", "memorial", "mainstream", "listen", "lavrov", "jersey", "islam", "irans", "investment", "insurance", "hopeful", "hiding", "here", "given", "gingrich", "fuel", "fan", "expert", "exit", "exactly", "eric", "enforcement", "duterte", "dirty", "diplomat", "deadline", "dark", "cuban", "critical", "credit", "corrupt", "confident", "buy", "briefing", "brave", "becomes", "beating", "april", "allowed", "agrees", "9", "\u201ca", "zone", "whether", "western", "warrant", "trust", "tower", "torture", "toll", "tim", "themselves", "that\u2019s", "tech", "taps", "sudan", "store", "someone", "silent", "sharpton", "selling", "seeking", "scam", "remain", "prove", "probes", "primary", "price", "position", "pathetic", "paid", "openly", "once", "nazi", "month", "medical", "me", "massacre", "male", "lower", "looking", "local", "links", "limits", "land", "kansas", "it\u201d", "islamist", "influence", "india", "incident", "hypocrite", "hunt", "he\u2019ll", "hezbollah", "harry", "happy", "guests", "god", "glorious", "gang", "games", "floor", "falls", "eye", "explodes", "episode", "employee", "dumb", "drive", "door", "donors", "domestic", "detains", "cross", "critics", "courts", "course", "coulter", "congresswoman", "commission", "closed", "classified", "ceasefire", "cause", "canadas", "brazils", "bombing", "bombers", "blows", "bipartisan", "berkeley", "audience", "arab", "among", "ago", "accusations", "account", "2015", "\u201cthis", "\u201ci\u2019m", "worked", "withdrawal", "willing", "we\u2019re", "went", "website", "venezuelas", "undercover", "true", "trudeau", "tom", "todd", "sued", "stands", "standing", "spokeswoman", "shutdown", "shoots", "several", "risks", "restaurant", "rescue", "rate", "raqqa", "privacy", "politico", "pledges", "photos", "paying", "opening", "online", "oklahoma", "officers", "nsa", "newspaper", "movement", "kurdistan", "killer", "kid", "kasich", "june", "juncker", "joint", "jackson", "irs", "instead", "increase", "hysterical", "humiliated", "hours", "heart", "happen", "handed", "guard", "graham", "everything", "emergency", "disabled", "date", "coup", "cost", "collapse", "cold", "closer", "claiming", "canadian", "building", "brags", "begins", "based", "backing", "arizona", "antigay", "ann", "amendment", "aims", "aim", "admit", "accuse", "60", "25", "16", "\u201cwhite", "\u201cnot", "\u2018white", "\u2018i", "zealand", "wounded", "wednesday", "wasn\u2019t", "vet", "unlikely", "truck", "trillion", "through", "threatening", "tests", "teachers", "stops", "stephen", "southern", "soldier", "sexist", "server", "serve", "sell", "search", "scheme", "scalia", "ross", "rock", "reveal", "rand", "proposed", "promise", "project", "problem", "presses", "president\u201d", "plot", "pledge", "pipeline", "oversight", "organizer", "nominees", "news\u201d", "network", "neil", "needed", "myanmars", "minimum", "mental", "memo", "medicaid", "maine", "lot", "liar", "legend", "lefty", "jets", "jesse", "jay", "invites", "interior", "innocent", "infrastructure", "indian", "including", "holding", "hitler", "historic", "helped", "harvard", "happens", "haley", "hacks", "graft", "governors", "gorsuch", "gone", "giuliani", "gaza", "gains", "flights", "fit", "feud", "fcking", "famous", "fair", "failure", "extrump", "expose", "explosion", "environmental", "doubles", "dollar", "doctor", "diplomacy", "dhs", "defeat", "damascus", "crossing", "considering", "congo", "confronts", "comedy", "coast", "civilians", "christians", "challenges", "cbs", "camp", "brilliantly", "both", "blood", "ball", "austrian", "appearance", "alleged", "again\u201d", "advance", "14", "\u2018fake", "writer", "within", "witch", "wars", "walk", "vile", "viewers", "vatican", "usbacked", "ugly", "trumpcare", "throw", "swiss", "susan"], "legendgroup": "", "marker": {"color": "#636efa", "size": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "sizemode": "area", "sizeref": 0.25, "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scattergl", "x": [0.10853249579668045, 0.08580254018306732, -0.595019519329071, -0.26565444469451904, -0.6843729019165039, 1.5763477087020874, 1.3963154554367065, 0.44315841794013977, -0.5435340404510498, 1.5819095373153687, -1.4967286586761475, 0.46779850125312805, 1.5935242176055908, -0.7325381636619568, -1.5020837783813477, -1.286515235900879, 0.8062139749526978, -2.4004299640655518, 0.5178616046905518, -1.4671974182128906, 0.255826473236084, 0.8761596083641052, 0.21454483270645142, 0.20814093947410583, -0.09823330491781235, 0.1730499565601349, 0.02491903305053711, 0.574141800403595, -0.8991837501525879, 0.7199859023094177, 0.018578389659523964, -0.6737945079803467, -0.7457684278488159, 0.05354558303952217, 0.07563856989145279, -0.011875947937369347, 0.6582908034324646, -1.1580982208251953, -0.1883695125579834, -0.056599028408527374, 0.33686697483062744, -0.9265515804290771, 1.0988489389419556, -0.26808199286460876, 0.5576059818267822, 0.6294360160827637, 1.5804201364517212, 0.07336562126874924, -1.0054199695587158, -0.14523158967494965, -1.2795745134353638, 0.36987632513046265, -0.15486110746860504, 1.0273205041885376, 0.010747491382062435, -0.2213646024465561, 0.352289080619812, 0.16311071813106537, -0.19201669096946716, -0.6060637831687927, -0.0685313269495964, 0.13073256611824036, -0.29766586422920227, -1.2148735523223877, 2.677982807159424, 0.315087229013443, 0.8863353133201599, -1.103235125541687, -0.7934840321540833, -4.465903282165527, 6.226966857910156, -1.2445335388183594, -0.8567007184028625, 0.9393842816352844, 0.5941048860549927, -0.3587152659893036, 0.37188786268234253, 0.09092754870653152, -0.4397923946380615, -0.36366957426071167, 0.04067511484026909, -0.02832365408539772, -0.086835116147995, 0.19461657106876373, -1.5178500413894653, -0.025190846994519234, 0.06186743825674057, -1.083457589149475, -1.2105145454406738, -0.37290292978286743, 0.6650791764259338, -0.27794012427330017, -0.3634788990020752, 0.41702383756637573, 0.7451339960098267, 0.04501519352197647, -0.3641090393066406, 0.24560287594795227, 1.74315345287323, -0.11684602499008179, 0.1641656905412674, -0.5186404585838318, 0.4472590684890747, 0.016474932432174683, 0.35167598724365234, -0.7882333993911743, 0.6009918451309204, 0.6588174700737, -1.929498553276062, 0.21731369197368622, -0.33855393528938293, 1.221988320350647, -0.38123685121536255, -0.08928760141134262, -1.553830862045288, 0.10819991677999496, -0.8113833069801331, 0.11002769321203232, 0.16897691786289215, -0.059606265276670456, 0.7919260859489441, 0.0071811070665717125, 1.487830638885498, 0.19989050924777985, 0.1495015025138855, 2.2198052406311035, -0.9978812336921692, -0.42647647857666016, 0.11031534522771835, -0.4807477593421936, 0.3119249939918518, -0.15436692535877228, 0.5545762777328491, -0.5253396034240723, -0.3005969524383545, -0.688368558883667, 0.49661785364151, -0.015236863866448402, -1.2577614784240723, -0.18249109387397766, -3.23185396194458, -0.5291119813919067, -0.09094707667827606, 1.572576642036438, 0.3205677568912506, -0.10295223444700241, -0.10940685868263245, 0.04210345074534416, 0.09021368622779846, 0.4685952365398407, -0.39029747247695923, 0.27646857500076294, -0.6959943175315857, 1.6813050508499146, 0.7967219948768616, 0.43751147389411926, -0.8179620504379272, 0.38748839497566223, 0.11049580574035645, 0.21606691181659698, -1.2578847408294678, -1.3562613725662231, -1.477259635925293, -0.4722098708152771, 0.06577924638986588, 0.1263931840658188, 1.996509075164795, -0.8513541221618652, 0.5840840935707092, -0.5063985586166382, -1.0601757764816284, -0.7084102630615234, 0.8630207180976868, 0.0616062693297863, -1.5622227191925049, 0.5257375240325928, 0.027224162593483925, 0.0196374524384737, 0.29517507553100586, 0.160971999168396, 0.7751942276954651, -0.9382906556129456, -0.5078479647636414, -0.3609727621078491, 1.2494335174560547, 0.4551030695438385, -0.28424251079559326, 0.093841552734375, 0.35414716601371765, 0.6384696960449219, -0.2628669738769531, 0.25450965762138367, 0.003110341727733612, 0.5009394288063049, 0.11489848047494888, -0.7116708159446716, -0.9757775664329529, 0.16585469245910645, 0.7621387839317322, -0.6818365454673767, 0.7074509859085083, 0.2909756898880005, -0.9610910415649414, 0.5930211544036865, -0.6539602875709534, -0.08913392573595047, -0.15310576558113098, 0.6100023984909058, 0.07124467939138412, -0.25287577509880066, -1.5842357873916626, 1.0064029693603516, -0.250196635723114, 0.26055940985679626, 1.193625807762146, 0.12288975715637207, -0.5638579726219177, -0.20207849144935608, 1.0975195169448853, -0.9152270555496216, 0.1347990483045578, -0.3029593527317047, -0.30649811029434204, -1.2946271896362305, 1.2807587385177612, 0.2702440023422241, 0.47289687395095825, -0.46146905422210693, 0.22152191400527954, 0.44141674041748047, 0.42093026638031006, 1.6349971294403076, 0.3595845699310303, 0.10489682853221893, -0.134125217795372, 0.6564489603042603, 0.2509561777114868, -0.6481711864471436, -0.5723164081573486, -0.015902018174529076, 0.01083026546984911, 0.404986172914505, 0.14601927995681763, 0.6154072284698486, -0.49743860960006714, 0.5971481800079346, -0.08357320725917816, 1.5092300176620483, -0.2797890901565552, 1.262183427810669, 0.3612324297428131, 0.024809103459119797, -0.5411171317100525, 0.2935296893119812, -0.2967241406440735, -0.08721307665109634, -0.3826655447483063, -0.19570016860961914, -0.3834294378757477, -0.12228734791278839, 0.7152004241943359, -2.0462839603424072, -1.8502776622772217, -0.24697324633598328, 0.19406642019748688, 0.18927417695522308, -0.5048498511314392, 0.008004548028111458, 0.45279937982559204, 0.15256044268608093, -1.095984935760498, -0.3558160662651062, 0.004486645571887493, 0.5481082201004028, 0.17314699292182922, 0.5659512877464294, -0.16607801616191864, -0.9839419722557068, 0.1318707913160324, -0.4451400339603424, -0.19487646222114563, 0.35270723700523376, -0.6360273361206055, -0.2081090360879898, 0.09514117240905762, -0.6166309118270874, -0.3135894238948822, 0.11323992162942886, 0.31350281834602356, -0.4636545479297638, 0.1751042753458023, 0.5187624096870422, -1.2574595212936401, 0.8346218466758728, -0.19144637882709503, -0.2532036304473877, -0.12167371809482574, 0.40215829014778137, 0.4550902545452118, -0.307947039604187, 0.5015969276428223, 0.2962146997451782, 0.7686026692390442, 1.131481409072876, 0.8510773777961731, -0.5208505392074585, 0.10570406168699265, 0.15551796555519104, 0.800566554069519, -1.3933857679367065, 0.435913622379303, -0.7307736277580261, 0.802399218082428, -0.490660160779953, 0.09278729557991028, -0.0606234036386013, -1.3928308486938477, -0.3192354440689087, 0.4024440050125122, -2.491187810897827, -0.5955834984779358, 0.11368782818317413, 0.5631248950958252, -0.9099401235580444, 0.24416542053222656, 0.27374979853630066, 0.17955583333969116, -0.7028102874755859, -0.7991148233413696, 0.8354461789131165, -0.19162650406360626, -0.11246237903833389, 0.39410001039505005, 0.7387018203735352, -0.19355030357837677, -0.6609055399894714, -0.5412724018096924, -0.7912214398384094, 1.5019992589950562, 0.021793082356452942, -1.0438629388809204, -0.04025736823678017, -1.2411210536956787, 1.3507630825042725, -0.014254219830036163, 1.6617006063461304, -0.11502519249916077, 0.2506655752658844, 0.2183174192905426, -1.7287945747375488, -0.3755362331867218, 0.2489517778158188, -0.8772085309028625, 1.1602795124053955, -1.8635023832321167, 1.3789491653442383, 2.2157440185546875, 0.6315076351165771, -1.4405959844589233, -0.28501763939857483, -0.18244370818138123, -0.01259792409837246, -0.023701118305325508, -0.1399005949497223, 0.6033234596252441, -1.4515414237976074, 0.15465401113033295, -0.1105256900191307, 0.0327359214425087, 0.7329627871513367, -1.437239646911621, -0.5293930768966675, 0.6139112710952759, -0.28546878695487976, -0.8052639961242676, 0.4968259036540985, -0.0204207394272089, -0.18921852111816406, -0.031799107789993286, -0.32585614919662476, -0.8663274049758911, -0.8584539294242859, 0.5988550782203674, -0.16719907522201538, 0.35087043046951294, 1.3237196207046509, 1.0594528913497925, 2.5107758045196533, 0.5241411328315735, 0.332841157913208, 0.20954285562038422, 0.33373382687568665, -1.817380666732788, 0.5655254125595093, 1.5718870162963867, 3.0185930728912354, -0.31919190287590027, -0.12833966314792633, 0.20398269593715668, 0.5293801426887512, -1.2475961446762085, -0.10959379374980927, -0.00137103081215173, 0.2646276354789734, -0.15279021859169006, 0.21332313120365143, 0.07608753442764282, 0.5458296537399292, 0.21137703955173492, -0.19755475223064423, -0.32399454712867737, 0.9154013991355896, -0.4572671949863434, 0.40195009112358093, -0.7546630501747131, 2.0559608936309814, -0.5603755116462708, 0.5391075015068054, 0.028514914214611053, 0.06293618679046631, 1.4514888525009155, 0.24947892129421234, 0.4237125515937805, 1.813883900642395, 0.8746296763420105, -0.05736980214715004, -0.4091109037399292, -1.1580450534820557, 0.3642630875110626, 0.5400398969650269, -0.23048025369644165, -0.45551881194114685, -1.1242872476577759, 0.24866211414337158, -0.6075488924980164, 0.31394436955451965, 0.037998780608177185, -0.4229165315628052, 0.11060696840286255, 0.4071309566497803, -0.9180675148963928, 1.4443014860153198, 0.8580141663551331, 0.952850341796875, -0.1724577248096466, -1.2176792621612549, -0.2400563806295395, 0.7848266363143921, -0.7532876133918762, 0.5190520882606506, 0.5478364825248718, 0.30676954984664917, -0.30554577708244324, -0.47352275252342224, 2.1185059547424316, -0.3719429075717926, 0.24602413177490234, 0.23885364830493927, 0.4255383312702179, -0.8865640759468079, -0.4010312855243683, 0.8915235996246338, 0.4209797978401184, 0.8724349141120911, 0.34597843885421753, 0.47362610697746277, -0.6181000471115112, -0.7240404486656189, -0.22796615958213806, -0.5319744944572449, -0.6388657093048096, 0.0992160215973854, -0.49012982845306396, -0.22670654952526093, -0.64963698387146, 1.57415771484375, 0.3699966371059418, 0.19542871415615082, -0.1357378363609314, -1.8823754787445068, -0.8722781538963318, 0.8170514106750488, 0.27355992794036865, -0.09413491189479828, -1.3520478010177612, -0.04857345297932625, -0.47654134035110474, -1.2212961912155151, -0.2839655876159668, 1.3276807069778442, -0.9584716558456421, -0.7412675619125366, 0.9146109819412231, 0.484385222196579, -1.073879361152649, 0.2531069815158844, 1.0974317789077759, -0.5724943280220032, 0.0978183001279831, 0.3538035452365875, 0.06250807642936707, 1.4481055736541748, -1.5531351566314697, 0.12180056422948837, 0.2410537749528885, -0.02977736108005047, -0.7731025815010071, -0.9202333092689514, 0.06810957938432693, 0.2843003571033478, -0.1763087511062622, -1.872459053993225, 0.35196104645729065, 0.09405926614999771, 0.7061887383460999, 0.3502967357635498, 0.2731595039367676, 0.10279297828674316, 0.5943112969398499, -1.3223124742507935, 0.27347344160079956, -0.19309481978416443, -0.3995954692363739, -0.12697555124759674, 0.18502439558506012, -0.6057484149932861, 0.13974368572235107, -0.8453507423400879, -0.2948397994041443, 0.11161767691373825, 1.2821470499038696, -0.9395732879638672, 3.403775215148926, -0.005831940099596977, 0.326835960149765, -0.3021725118160248, 2.0294888019561768, 0.5348012447357178, -1.1611531972885132, -1.1123160123825073, -0.5604498982429504, 0.8521620035171509, -0.12074786424636841, -0.300947904586792, -1.6056151390075684, -0.16015909612178802, -0.27085694670677185, -1.0872993469238281, 1.1173652410507202, 0.5176379680633545, 1.755002737045288, -0.6164785027503967, -0.5149005651473999, 1.0545430183410645, -0.7446855306625366, 1.5800749063491821, 0.69219970703125, -0.03481020778417587, -0.6074298024177551, -0.4063528776168823, -0.9999580979347229, -0.7864211201667786, 1.360310673713684, 0.5339862704277039, -0.07065358012914658, -0.30825215578079224, -1.1916277408599854, 0.40341731905937195, -0.6873779892921448, 0.08601254969835281, 0.3766595423221588, 0.0704040378332138, 0.09345614910125732, -0.8695343136787415, -0.6623916625976562, -0.349046528339386, -0.010188749991357327, 0.12306862324476242, -1.416496753692627, -3.7049546241760254, 0.6746257543563843, -0.07020419836044312, 0.11100616306066513, 0.4565178155899048, -0.25733432173728943, 1.4731696844100952, -0.7249196171760559, -0.14665749669075012, 0.8206098675727844, -0.9336216449737549, 0.1690724492073059, 1.200046420097351, 0.6235554218292236, -0.3489636182785034, 1.1751282215118408, -0.32487615942955017, 0.587888240814209, -0.3329036235809326, -0.13221250474452972, -0.6324548721313477, 0.7844599485397339, -0.25817573070526123, 0.3218570351600647, 1.9588062763214111, 0.4168594479560852, 0.37711024284362793, -0.8103733062744141, 0.37355202436447144, 1.1735798120498657, -0.5498802661895752, 0.4925035238265991, 0.434846431016922, 0.0888625904917717, -0.20473358035087585, 0.07703929394483566, 0.2995043396949768, 1.7372642755508423, -1.5285089015960693, 0.09912922978401184, -1.8170350790023804, 0.13496346771717072, 1.2228977680206299, -0.7904798984527588, 0.5498553514480591, 0.5439239144325256, -1.2483255863189697, 0.37727662920951843, 0.28463444113731384, -2.3850417137145996, -0.35820063948631287, -0.5578227043151855, 0.03611404448747635, -0.3498871922492981, -0.581024706363678, -0.712262749671936, -0.1869669109582901, -0.18563824892044067, -0.5253455638885498, -1.186228632926941, 0.4576718211174011, -0.32095572352409363, -1.3097259998321533, 0.35018664598464966, -1.2820935249328613, -0.47286421060562134, 1.2865943908691406, -0.16588596999645233, 0.15338557958602905, 0.8556768894195557, 0.4292784333229065, -0.4172113537788391, 0.09884864836931229, 0.02299368754029274, -0.43592649698257446, -0.3991289734840393, -0.3754502534866333, -0.4649661183357239, -0.8155457377433777, -0.1897585242986679, -0.49275651574134827, 0.5012937188148499, -0.6744927167892456, -0.3260009288787842, 0.3112800717353821, -0.13548272848129272, 0.29532232880592346, 0.22334519028663635, -0.4579938054084778, -1.7739741802215576, -0.01828831061720848, -0.7804877161979675, -0.9454342722892761, 0.4008084833621979, -0.06914345920085907, 0.100702203810215, -1.370381474494934, 0.42733731865882874, -0.14614258706569672, -0.6540815830230713, 0.5267947912216187, -0.18471263349056244, 0.5345589518547058, -0.028286350890994072, 0.4148222804069519, 0.7978416681289673, -0.6177319884300232, -1.1788922548294067, 0.2274022400379181, -0.007819700054824352, 0.21197889745235443, 0.08023183792829514, 0.12219759821891785, -0.4850865602493286, -0.2645134925842285, -1.1070622205734253, 0.3744259178638458, -0.33120039105415344, 1.7953983545303345, 0.45686274766921997, -0.510219156742096, -0.18549078702926636, 1.1987577676773071, 2.430623769760132, -1.5475788116455078, -0.24405167996883392, 5.6296550610568374e-05, -0.09673097729682922, -0.012016581371426582, 0.8040761947631836, -0.22812457382678986, -1.4839212894439697, -1.3495575189590454, 0.07553298026323318, 0.6258680820465088, -0.557157039642334, 0.7844428420066833, -0.5822494626045227, -0.2414345145225525, -0.3776918649673462, 0.9418932795524597, -0.6141510605812073, -0.20609192550182343, -0.19203698635101318, 0.4430316686630249, -0.42197203636169434, 0.13819870352745056, 0.7726475596427917, 0.34028568863868713, 0.6445469856262207, -0.2915985584259033, -1.5365833044052124, -0.16462944447994232, -0.23037083446979523, -0.05815969780087471, -0.5961663126945496, 0.558118999004364, -1.0088326930999756, -0.593412458896637, -0.558260440826416, 1.1961954832077026, 0.747018575668335, -0.5491055250167847, 1.058547854423523, 1.1449329853057861, 0.27547794580459595, -0.0988425463438034, 2.0585503578186035, -0.07731068134307861, -1.2113202810287476, 0.3210891783237457, 0.050034862011671066, -0.4272376596927643, 0.9662754535675049, 0.19750851392745972, -0.19004201889038086, -0.2708513140678406, -0.820426881313324, -1.2883317470550537, 2.4226083755493164, -0.015644701197743416, -0.41156455874443054, -1.5595136880874634, -0.8561323881149292, -0.18591710925102234, -0.554119348526001, -0.5619511008262634, -0.3979191482067108, -0.6955759525299072, 0.11813881248235703, 0.35927969217300415, -1.157762885093689, -0.20275390148162842, -0.3458816409111023, 0.7444211840629578, 0.326815128326416, 0.06902168691158295, -0.3033542335033417, 0.4219377040863037, -0.13059811294078827, -1.8051846027374268, -0.03508884459733963, -0.330488383769989, -0.7928482890129089, 0.9560211300849915, -0.2957642376422882, 0.24799825251102448, -0.38272804021835327, 0.11048945039510727, 0.12513643503189087, 1.095508098602295, -0.0489971749484539, -0.3294253349304199, 0.17800819873809814, -0.7085176706314087, 0.5140628814697266, -0.4608326852321625, 0.2451527714729309, -0.4919630289077759, -0.1702769696712494, 1.3857457637786865, 0.6295120716094971, 0.7774156332015991, 0.32607391476631165, 0.5519585013389587, 0.04900297522544861, 0.743135392665863, 0.38150471448898315, 2.1506612300872803, -0.2279539555311203, 0.1794513612985611, -0.5762865543365479, -0.9029165506362915, 0.8944271802902222, 1.090757131576538, -0.8665751814842224, 1.1217141151428223, 1.7343697547912598, 0.42300960421562195, 0.8602917194366455, -1.3914382457733154, -1.2429746389389038, -0.47600114345550537, -0.050882577896118164, -0.5054362416267395, -0.12327639758586884, 0.6063810586929321, 1.268406629562378, -0.40970349311828613, -0.490923136472702, -0.11623454093933105, 0.09510677307844162, -0.14816465973854065, 0.5199183821678162, -0.7563702464103699, 1.4255884885787964, -0.01814049668610096, 0.171820268034935, -1.754006266593933, 0.2544902265071869, 1.0180549621582031, -0.0817657932639122, -0.5735135674476624, -0.1045486107468605, -0.0878722295165062, 0.6616145968437195, 0.6800389289855957, 0.43215563893318176, -1.4317896366119385, -0.7197497487068176, -0.28729209303855896, -0.13395480811595917, -0.06872983276844025, 0.4769003987312317, 0.6551293134689331, -1.6131609678268433, 0.3243427872657776, -0.6421250700950623, 0.3018278479576111, 0.2110339254140854, 0.8669079542160034, 1.063392996788025, -0.4138036370277405, 0.62503981590271, 0.06860577315092087, 0.826778769493103, -1.092253565788269, 0.5916938781738281, -0.6249152421951294, -0.7478711009025574, 0.1523955762386322, 0.7149008512496948, 0.023974185809493065, 0.8706706166267395, 0.6133544445037842, -0.2651275098323822, -1.8167906999588013, -0.06245477497577667, 0.09111625701189041, 0.38368159532546997, 0.5078201293945312, -0.3352988660335541, -1.7496529817581177, -0.055193014442920685, 0.23448331654071808, -0.8592340350151062, 0.8336392641067505, -0.4851098358631134, 0.056059833616018295, -0.9477143287658691, -0.6748981475830078, 1.211570382118225, -0.5066605806350708, -0.9036651253700256, 0.8523120880126953, -0.06177619844675064, 0.48924773931503296, 0.026393834501504898, -1.3344416618347168, 0.3372761011123657, -1.6232960224151611, -0.1623922735452652, -0.10789362341165543, 0.403135746717453, 1.0311532020568848, 0.2738104462623596, 0.7492890357971191, -0.8651549816131592, -0.92296302318573, -0.666284441947937, 0.7334824800491333, -0.7274381518363953, 2.5398409366607666, -1.4502774477005005, 0.9147881865501404, 0.7265236377716064, 1.099255084991455, -0.6403247117996216, -0.8362205028533936, 1.3954212665557861, 1.0718251466751099, -0.22900256514549255, -1.4228500127792358, 0.05838335305452347, 0.2696022391319275, 0.5125514268875122, 0.31977540254592896, 0.09001418203115463, 0.12065233290195465, -0.3167397379875183, 0.28118565678596497, -0.9216878414154053, -1.4739484786987305, 0.15045568346977234, 0.5742012858390808, -0.027093328535556793, -0.3062036633491516, 0.5828420519828796, 0.7971813678741455, -0.46801522374153137, 0.05991286411881447, -0.20582714676856995, 1.3476630449295044, 0.40545785427093506, 0.8589689135551453, 0.30840957164764404, 0.5039058327674866, 0.27618858218193054, -0.018261170014739037, 0.13298800587654114, -0.7574266791343689, 1.0625356435775757, -0.7866717576980591, 0.29031115770339966, -0.5336620807647705, -0.03229338675737381, 0.7405995726585388, -0.6067447066307068, 0.7606717348098755, 0.7484664916992188, 0.09367882460355759, -0.2900148928165436, 0.10328102856874466, -1.5677367448806763, 0.21791712939739227, 0.5722524523735046, -0.7639229893684387, -0.34238120913505554, 0.35410869121551514, 0.4938688576221466, -0.24354702234268188, -0.33506280183792114, -0.8167463541030884, 0.9490305781364441, -0.8662253022193909, -1.6608433723449707, 0.34801894426345825, 1.137802004814148, 0.6090258359909058, 0.7407522201538086, 1.247422218322754, -0.8404220342636108, 0.11359795928001404, -0.9054461121559143, 0.7051456570625305, 0.3274053633213043, -0.9301874041557312, 0.181187242269516, -0.717048168182373, 0.6238449215888977, -1.6015665531158447, -0.6629459261894226, 0.13688474893569946, -1.8748115301132202, -0.010107995010912418, -0.5319086909294128, 0.6934376358985901, 0.9919029474258423, 0.6106292009353638, -0.024279724806547165, 0.6831868886947632, 1.217244029045105, -0.2815932333469391, -0.27143773436546326, -0.3535255789756775, 0.4562384784221649, 0.12264934182167053, 0.10663031041622162, -0.506806492805481, 0.20840522646903992, 1.0363945960998535, 0.7341281771659851, 0.2667681574821472, -0.9374743103981018, 0.726720929145813, 0.47934627532958984, 0.6605580449104309, 0.22350363433361053, -0.04252498224377632, -1.3224064111709595, -1.6243302822113037, 0.14931026101112366, 0.38212963938713074, 0.6879488825798035, 0.7615053653717041, -0.3596017360687256, -0.8974496126174927, -0.8991537690162659, -1.524911642074585, 0.840568482875824, 2.075464963912964, -0.02355911396443844, 0.1947227120399475, -0.09268075227737427, 1.3195655345916748, -0.6895686984062195, -1.1879552602767944, -0.3150378465652466, 0.5589554905891418, 0.8345425724983215, -0.26892703771591187, 0.10057676583528519, -0.9480116367340088, -0.7620832324028015, -0.440211683511734, -0.52154940366745, -0.48946821689605713, 0.1438392549753189, 0.6684768199920654, 0.7268588542938232, 0.09941641241312027, -0.6453702449798584, -0.1772843301296234, -0.258524090051651, -0.18521912395954132, 3.7997539043426514, -0.39397990703582764, 0.7312661409378052, 0.32360509037971497, 0.2768949270248413, -0.3985743522644043, -0.8826022744178772, 1.1986706256866455, -0.3657873868942261, -0.08400394767522812, 0.35911786556243896, -0.6258677244186401, -0.01258365623652935, -0.6758789420127869, -0.281849205493927, 1.1967577934265137, 1.7766274213790894, 0.28833651542663574, -0.3355308473110199, -0.561811625957489, -1.1786006689071655, -0.2526637017726898, 0.8562987446784973, 0.0014000125229358673, -0.4770090579986572, -0.1711507886648178, 1.2037492990493774, 0.7890779972076416, 0.6051012277603149, 0.3300558626651764, -1.0199912786483765, 0.12445336580276489, 3.042003631591797, 2.2995221614837646, -0.7434501647949219, -1.3836722373962402, 0.6806954741477966, -1.6914145946502686, 0.08237755298614502, 1.2901476621627808, -0.48364028334617615, 0.08669133484363556, 0.6810360550880432, -0.27445870637893677, -0.6449684500694275, -0.10306959599256516, 0.6015458106994629, 0.5255730748176575, -0.19024920463562012, -0.7848039865493774, -1.062391996383667, 1.7439649105072021, -0.4210691452026367, -1.7554079294204712, -0.8699761629104614, -0.4894733428955078, -0.30966559052467346, -0.02171557955443859, -0.9214286804199219, -0.5328715443611145, -0.5156885385513306, 0.07451820373535156, 0.47901594638824463, -0.0947432890534401, 2.7112321853637695, 0.5226417183876038, -0.17991937696933746, -0.5325750708580017, 0.601930558681488, 0.5095744729042053, -0.2698071599006653, -0.4451843500137329, -0.3395562469959259, 0.6472116708755493, 1.4504990577697754, -0.16820256412029266, 0.31760844588279724, 0.508392870426178, -1.0428760051727295, -0.1488162726163864, -2.880079984664917, 0.5782735347747803, -0.7274697422981262, -0.33223956823349, 0.13374827802181244, -0.6172988414764404, 0.45626014471054077, -0.3749687671661377, -0.632910430431366, -2.278869390487671, -0.41556406021118164, -1.3579893112182617, 1.9378148317337036, -0.7824280858039856, 0.1621416062116623, 0.6606506705284119, 0.5639244318008423, 0.5619367361068726, -0.1391853243112564, -0.2801590859889984, -0.7358548641204834, -0.8427125811576843, -1.5242489576339722, 0.1416977494955063, -0.6762914061546326, -0.7336358428001404, -0.07109588384628296, 1.9853014945983887, -0.5802928805351257, -0.33755913376808167, -0.38681840896606445, 0.60039883852005, 0.36947864294052124, -0.22697490453720093, 0.6565268635749817, -0.1650020182132721, -0.07624534517526627, 0.857954740524292, 0.6289268732070923, -0.13781845569610596, 0.09277674555778503, 0.11819779127836227, 0.3153723478317261, -0.19090010225772858, 0.6749540567398071, 1.617335557937622, 0.42085739970207214, -0.5620586276054382, -0.790239691734314, -0.2991030216217041, -0.052091680467128754, -0.9549670219421387, -0.6654506921768188, 0.24475416541099548, -0.07685218006372452, -0.8656688332557678, 1.0056525468826294, -0.5052831172943115, -0.4537806212902069, 0.19965407252311707, 0.08798491954803467, 0.4975616931915283, -0.6346400380134583, 0.2503403425216675, 0.6326773166656494, -0.7714961171150208, -0.4451027810573578, -0.5653973817825317, 0.680059552192688, 0.2050045132637024, 0.49855655431747437, -0.48148277401924133, 0.4657748341560364, 0.93183434009552, 0.723457932472229, -0.6270988583564758, 0.25142285227775574, -0.3822714686393738, 0.6370441317558289, 0.40156498551368713, 0.8402379751205444, 0.7512761354446411, 0.642296552658081, 0.5800878405570984, -0.22098477184772491, 0.20109210908412933, 1.4689611196517944, 0.2647404372692108, -0.6076220273971558, 1.6373188495635986, 1.1670373678207397, 0.578475832939148, 0.8372158408164978, 0.08458458632230759, -0.12155470997095108, -0.06598589569330215, -0.09145908057689667, 0.5732539296150208, 0.8650839924812317, -0.4369238615036011, -0.5918512344360352, 0.33484286069869995, -0.2102617770433426, -0.03376398980617523, 1.4592509269714355, 0.026345141232013702, -0.7840995788574219, 2.6301188468933105, 2.8511416912078857, -1.0604820251464844, -0.4841366410255432, -2.1766791343688965, -0.40709832310676575, -0.835304856300354, -0.12112586200237274, -0.6172603368759155, -0.8561301231384277, -0.39448899030685425, 0.5874150395393372, 1.1374396085739136, -0.07702866941690445, 0.7412201762199402, 0.8870905637741089, -0.7217336893081665, -0.486860066652298, 0.9063352942466736, 0.49645769596099854, 0.27522486448287964, 0.3457772135734558, 0.08533737808465958, 0.9589686989784241, 0.5955907106399536, -0.6634442806243896, -0.6442293524742126, 0.12172476202249527, -0.8722397089004517, 0.33769312500953674, 0.667548656463623, 0.02458193153142929, -0.8852293491363525, 0.23815661668777466, 0.8320181965827942, 0.3343700170516968, -0.7203182578086853, -1.382561206817627, 1.034301996231079, 0.20959337055683136, -0.002866534749045968, 0.9071168899536133, -0.9264151453971863, -0.1212579533457756, 0.24895749986171722, -0.4059291183948517, 0.0964643582701683, 0.7273108959197998, -0.3640553653240204, -0.11191460490226746, 0.1376868337392807, 0.816286563873291, -0.0023683193139731884, 0.11080753803253174, -0.6140222549438477, -0.1966305524110794, -0.7505242228507996, -1.4996665716171265, -0.5533896088600159, 0.4970667362213135, 0.7481577396392822, -1.542559266090393, -1.5005608797073364, -0.19966985285282135, 0.8082573413848877, 0.8823131918907166, 0.6463277339935303, -0.31810399889945984, -0.16643202304840088, 0.07928923517465591, -0.5786200165748596, -0.7867560386657715, 0.6305024027824402, 1.1519582271575928, -0.7811359167098999, -0.9564219117164612, 0.4775463044643402, -0.7411101460456848, -0.4884103536605835, -0.0018513596151024103, 0.10925699770450592, 0.43749305605888367, 0.4518899619579315, 0.7691861391067505, -0.4770844578742981, -0.4981682002544403, -0.5487269163131714, 0.7410722970962524, -1.223258376121521, -0.6776519417762756, -0.683056116104126, -0.8817687034606934, 1.684826135635376, -1.4085415601730347, 0.08572512120008469, 0.39836814999580383, -0.19420526921749115, 0.06974490731954575, -1.377020001411438, 0.18560992181301117, -0.8105486631393433, 2.3631858825683594, -0.7885881662368774, -0.6113560795783997, -0.31162089109420776, 1.2884960174560547, 1.3966610431671143, -0.2951028048992157, 0.6389310359954834, 0.32464921474456787, 0.17348457872867584, 0.15688608586788177, 0.5098845958709717, -0.37502673268318176, -0.26776668429374695, -1.0290666818618774, -0.22513122856616974, -0.4330521821975708, 0.8506754040718079, 0.29533258080482483, -0.26070067286491394, 0.48628902435302734, 0.7886435389518738, 0.47287026047706604, -0.5306957364082336, -0.8939341306686401, -1.0935100317001343, -0.18508177995681763, -0.8072431087493896, -0.21853914856910706, 1.0072084665298462, 0.16118735074996948, 0.33973702788352966, -0.3980477750301361, -0.12736032903194427, -0.12589304149150848, 0.7270156741142273, -0.8376047611236572, -0.15728504955768585, -0.566667914390564, 1.2013444900512695, 0.5452389121055603, 0.6627424955368042, -1.6293420791625977, -0.035052187740802765, 0.2747415602207184, 0.049252890050411224, 0.08009664714336395, 0.11042453348636627, 0.8500069379806519, 0.8878228664398193, -0.2429097592830658, 0.1736876219511032, -0.6344724893569946, -0.47702816128730774, 0.31237152218818665, 0.4318760335445404, 1.3614588975906372, -1.8049381971359253, 0.1315382719039917, 0.6197158694267273, -0.16268226504325867, 0.564935028553009, -1.971078872680664, 0.6905184388160706, 0.17687378823757172, 0.2641197144985199, -0.5455681085586548, -0.08027060329914093, 0.09107637405395508, -0.32807281613349915, -0.6264021396636963, -0.10102840512990952, 0.48338043689727783, -0.8351181149482727, -1.4672222137451172, -1.642611026763916, 0.5832000374794006, 0.9975906014442444, -0.0834437906742096, -1.3378934860229492, -0.028293538838624954, 0.05193447694182396, 0.17042042315006256, 0.08813649415969849, 0.568398118019104, -0.5374714732170105, -0.4673255383968353, 0.6171882748603821, -0.09432948380708694, -0.5712072849273682, -2.0471878051757812, 0.5210167765617371, -1.3073257207870483, -0.7257753014564514, 1.3214137554168701, 0.6616422533988953, -0.3362220823764801, -1.1692111492156982, 0.271005779504776, -0.19839146733283997, 0.3894781768321991, 0.5555629730224609, -0.045421820133924484, 1.0648995637893677, -0.35935020446777344, -0.42693936824798584, -0.763817310333252, 0.07318693399429321, 0.32857412099838257, 0.0027396108489483595, -0.1312597393989563, 1.1131287813186646, 0.19693797826766968, -0.31603530049324036, -0.45953676104545593, -0.5301504731178284, -0.27320945262908936, 0.7259747982025146, -0.6660387516021729, 0.10107264667749405, 0.045548178255558014, -0.9425158500671387, 0.22162850201129913, -0.06477005034685135, 1.051345705986023, 0.5275196433067322, 1.0207237005233765, 0.5206325054168701, 0.31480786204338074, -0.8785048127174377, 0.07457306236028671, -1.8617479801177979, 0.4581963121891022, -0.15525774657726288, 0.09816652536392212, -0.19591756165027618, -0.6546328067779541, -0.1145922914147377, 0.32947808504104614, 0.15342332422733307, -1.2885063886642456, -0.4648919105529785, 0.3316296637058258, 0.8030396699905396, 0.7992070913314819, -0.09032171219587326, -0.5315976142883301, 0.08531485497951508, -0.1463693380355835, 1.1175867319107056, -0.5590866208076477, 0.4753705561161041, 0.16294105350971222, 0.9818598628044128, 0.10302214324474335, 2.0836448669433594, 0.043156977742910385, -1.6961443424224854, 2.4225921630859375, 0.6153663992881775, -1.412153720855713, 0.23715968430042267, -0.7315666675567627, 0.827951192855835, 0.8374806046485901, 0.10274264216423035, 0.3907535970211029, 0.5346124172210693, -0.43472805619239807, -1.6328158378601074, -1.399717092514038, -0.015576091594994068, -1.3620847463607788, 0.06512288749217987, -1.1226019859313965, 0.07701840251684189, -0.8153332471847534, -0.96136075258255, 0.5466617941856384, -1.257820963859558, -0.13726723194122314, 1.0041909217834473, 0.4355103075504303, -1.3318978548049927, 0.06323723495006561, 0.5575942993164062, -1.6431856155395508, 0.14671175181865692, 2.3646178245544434, 0.7086304426193237, 0.06425492465496063, 0.08639159053564072, -0.2965726852416992, 0.8993450999259949, 0.7831333875656128, -1.0377410650253296, 0.9919415712356567, -0.13116355240345, -0.43394458293914795, 0.10383464395999908, -0.17372941970825195, 0.6916689872741699, 0.15895548462867737, -0.30542320013046265, 0.022243032231926918, -0.642507791519165, -0.03883153945207596, 0.49198833107948303, -0.2600216567516327, -0.03014126420021057, 0.9705025553703308, -2.965860605239868, -0.3402726650238037, 0.1551872044801712, 0.7725998163223267, -1.382157564163208, 0.20235562324523926, 0.0006106705986894667, -0.3497784435749054, 1.172162652015686, -0.8097914457321167, -1.0147382020950317, -0.04291658103466034, 0.753170907497406, 2.047804117202759, -0.9267036318778992, 0.03907962888479233, 1.046128749847412, 1.1682944297790527, 0.1476762294769287, -0.3768446445465088, -0.26160863041877747, -0.3203524351119995, -0.025331661105155945, -0.7931978702545166, -0.5904976725578308, 0.142014741897583, 0.5617850422859192, -0.1470262110233307, -1.3624310493469238, -0.023645050823688507, -2.467351198196411, 0.295813649892807, -0.5163102746009827, -0.693892240524292, -0.050906676799058914, 0.07013632357120514, 0.6673898100852966, -0.2818199694156647, 1.3329135179519653, -0.20963725447654724, 0.7783001065254211, 0.9479053616523743, 0.21998775005340576, 0.29492565989494324, 0.4777040183544159, 0.4432281255722046, 1.2252235412597656, 0.7825272679328918, 1.0866708755493164, -1.0086959600448608, 0.15530924499034882, 0.2710579037666321, 0.4201247990131378, -1.0964510440826416, 0.1641516238451004, 0.9582964777946472, -0.5861950516700745, -0.5040794014930725, -0.20490539073944092, 0.12210440635681152, -0.30754783749580383, -1.088144063949585, 0.4271189272403717, 0.06793094426393509, -0.5149122476577759, -0.7261738777160645, -0.5555879473686218, 0.7397823929786682, -0.2800576388835907, -0.7689306735992432, -1.2045193910598755, -0.10659954696893692, 0.07122103124856949, -0.18136827647686005, -1.001859188079834, 0.767137885093689, 0.5013339519500732, -0.920967161655426, -0.38468530774116516, -0.8799922466278076, 0.4626978039741516, -0.8893715143203735, 0.10807846486568451, 0.1565760374069214, -1.7527521848678589, -0.6990581750869751, -0.8973480463027954, -0.052008047699928284, -0.566992998123169, 1.7416775226593018, 0.4354712665081024, -0.9857268929481506, -1.443253517150879, 0.4803614616394043, 0.15782514214515686, 0.6046146154403687, 1.7820796966552734, 1.3348617553710938, -0.06628537178039551, 0.6078628897666931, -0.9669989943504333, -1.0978193283081055, -0.3304089903831482, -0.6245642900466919, 0.8972036242485046, -0.35018202662467957, -0.41640162467956543, -0.5274966359138489, -1.3122841119766235, 0.6347891688346863, -0.4082261621952057, 0.7252897024154663, -0.3724832236766815, 0.1953606903553009, -0.48857757449150085, 0.848303496837616, 1.1508033275604248, -0.27552172541618347, -0.016498759388923645, -1.4542295932769775, 0.0920070931315422, 0.7414631247520447, 1.67290198802948, -1.1628921031951904, 0.2615397870540619, -0.5399981141090393, -1.0464617013931274, -0.7113175988197327, 0.37012380361557007, 0.13226395845413208, 0.13979864120483398, -0.3404257297515869, 1.4771560430526733, 0.37520721554756165, -0.15665698051452637, 0.6510282158851624, 0.7738985419273376, -0.17828375101089478, 1.201287865638733, -0.5911484956741333, 0.13344818353652954, 0.14182260632514954, 0.42948123812675476, 0.18560467660427094, 0.6184486746788025, -0.17948707938194275, -0.8405765891075134, 0.39555078744888306, -1.156104326248169, -0.09296988695859909, 1.1414127349853516, -0.37003085017204285, 0.13525255024433136, -0.7746344208717346, 0.8395620584487915, -0.6641908288002014, 0.3530929982662201, -0.5270712971687317, -0.893770158290863, -0.014528677798807621, -0.6598929762840271, -0.8901056051254272, 0.41432639956474304, 1.459924340248108, 0.2102522999048233, 0.8375799655914307, -0.0577392615377903, -0.2263817936182022, 0.4449382722377777, 0.42474672198295593, 0.7571197152137756, 0.6701164841651917, 0.6007965803146362, 0.36103203892707825, -0.06028274446725845, 0.13655827939510345, 0.7040625810623169, 0.4491592049598694, 0.7318138480186462, -0.12012936919927597, -0.49102264642715454, 0.027410775423049927, 0.09153222292661667, 0.6080121994018555, 0.393814355134964, 0.2053905874490738, 0.38717254996299744, 0.09504954516887665, -0.037540633231401443, -0.0037528849206864834, 0.6207935214042664, 0.1914496123790741, -0.1299162209033966, 0.30747565627098083, -0.3435933589935303, -1.1318293809890747, -0.47609564661979675, -1.1288443803787231, 0.09468652307987213, -0.43844640254974365, -0.676709771156311, 0.4515050947666168, 0.36300671100616455, -1.006935954093933, 1.245877981185913, 0.5928341746330261, -1.2450417280197144, -0.6290327310562134, -0.506914496421814, 0.7433306574821472, 0.9650868773460388, 0.43844592571258545, 0.08165021985769272, -0.268189013004303, 1.0124647617340088, 0.5528764724731445, 0.27358242869377136, 0.3224331736564636, -0.5543627738952637, 1.0671571493148804, -0.2056693434715271, 0.6557450890541077, 0.07004237920045853, 1.3080071210861206, 0.3204704821109772, 0.8528927564620972, 0.22452573478221893, 0.1853541135787964, 0.11182072013616562, 0.16506846249103546, 0.2977871298789978, 0.7575985789299011, -0.9764997363090515, 0.16345112025737762, -1.5255999565124512, 0.884053647518158, 0.7573281526565552, -0.9080296754837036, -1.1732478141784668, -0.31647053360939026, -0.7394486665725708, -1.3510777950286865, 0.5099192261695862, -0.7977240085601807, -0.21469542384147644, 1.3296184539794922, 0.33762702345848083, 0.6917470097541809, 0.012209543026983738, 0.6138870716094971, -0.16792234778404236, 1.6987111568450928, 0.5390695929527283, -0.4313929080963135, 0.20194362103939056, -0.546156108379364, 0.09376081824302673, -0.11876369267702103, -1.2421925067901611, 0.3699502944946289, 0.8916968107223511, -0.2730225622653961, -0.6639316082000732, -0.13607485592365265, 1.5260852575302124, 0.2897263467311859, 1.295027256011963, -0.28065603971481323, -0.3962397277355194, 0.4124622941017151, -0.14137153327465057, 1.2038384675979614, 1.1130023002624512, 0.5028702020645142, -1.0382291078567505, -0.4052521884441376, -0.048372168093919754, -1.255342960357666, -0.4281465411186218, 0.4833917021751404, 0.8509969711303711, -0.0022712205536663532, 0.5593193173408508, 0.09719789773225784, -0.13237103819847107, -0.8944761157035828, 0.7401747107505798, -0.05860911309719086, 0.5057153701782227, 1.5202947854995728, -0.43669068813323975, 0.22799146175384521, -0.050957635045051575, -0.13534344732761383, -1.4218087196350098, 0.45975691080093384, 0.34180644154548645, 0.013527706265449524, -0.2421715259552002, 0.018759872764348984, 0.7140626907348633, -0.8553748726844788, -0.11167418956756592, -0.23742520809173584, -0.10651160776615143, 0.5263093113899231, 0.7344875335693359, -0.8300191760063171, -0.0014707647496834397, 0.038750436156988144, 0.5649263858795166, 0.18328909575939178, 0.11855243146419525, 0.13046197593212128, 0.0798356831073761, 0.5122244954109192, 1.3313233852386475, 0.015871994197368622, 3.052835464477539, 0.10550176352262497, -0.14765988290309906, 0.11137057095766068, -0.05137643590569496, 0.09161621332168579, -0.24963004887104034, -1.6480621099472046, 1.120712161064148, 0.2600690424442291, -0.1419784277677536, -0.22606876492500305, 0.3709210455417633, -0.26585668325424194, -0.7991904616355896, 0.33433327078819275, 0.2926693856716156, 0.22530363500118256, 0.4939056634902954, -0.49926868081092834, -0.015377186238765717, 0.8829034566879272, -0.3514310121536255, -0.22158750891685486, -0.36697155237197876, 0.49383124709129333, 0.13854241371154785, 0.14862117171287537, 0.03237446770071983, -0.03269931674003601, 0.18247616291046143, -0.20639750361442566, -0.37833893299102783, -0.339459627866745, 0.4691232740879059, -0.5662494897842407, 0.6056345701217651, 0.11233488470315933, -1.031227707862854, -0.28266090154647827, 0.39651504158973694, 0.36841174960136414, 0.08583752065896988, -0.4108820855617523, -0.05246487259864807, 0.08245836943387985, 0.24517041444778442, -0.4418184757232666, -0.5351903438568115, -0.5946181416511536, -1.2011466026306152, 0.4967837631702423, -0.5311388969421387, -0.11661305278539658, 0.3078326880931854, 1.0157055854797363, -1.2976266145706177, 0.1406194418668747, -0.09829305857419968, 1.0392494201660156, 0.2641378939151764, -0.3881186544895172, -0.7112454175949097, 0.12738509476184845, 0.10591480135917664, -0.8190425038337708, 0.15034830570220947, -1.0281877517700195, 1.7742884159088135, 0.07582944631576538, 0.022901995107531548, -0.7342191934585571, -0.13100160658359528, 0.7301294207572937, -0.548477292060852, 0.4493710696697235, -0.4618149995803833, -0.6159397959709167, -1.0062819719314575, -0.17271777987480164, -0.06687664240598679, -0.1228829026222229, 0.10626029968261719, -0.08114447444677353, 0.8156228065490723, 0.1834106296300888, -0.3476629853248596, 0.04683910310268402, -0.4628514349460602, 0.30892619490623474, -0.34176504611968994, 0.36906898021698, 0.7304201722145081, -0.6301587820053101, -0.6800653338432312, 0.1678110659122467, -0.6891829371452332, 0.05205608531832695, 0.6578736305236816, 0.36168840527534485, -0.011037174612283707, 0.9589394330978394, -1.3167986869812012, -0.09305494278669357, -0.9699594974517822, 0.7075595855712891, -0.23574583232402802, -1.0193952322006226, 0.9659245610237122, -0.20880156755447388, -0.7117473483085632, 0.12987156212329865, 0.6757246851921082, -0.9116454720497131, 0.1999659687280655, 0.19247174263000488, -0.7776060700416565, 0.45847392082214355, 0.2378002107143402, 0.3465197682380676, -0.8384026885032654, -0.21898266673088074, 0.9615738987922668, 0.3924199044704437, -0.2815497815608978, -0.9472655653953552, -0.8804638981819153, 0.5280690789222717, 0.32522740960121155], "xaxis": "x", "y": [-0.009206099435687065, -0.3105482757091522, -0.19932129979133606, -0.037148021161556244, 0.11111969500780106, -0.1719881147146225, -0.4591539800167084, -0.32855358719825745, -0.22810997068881989, -0.06442984193563461, -0.34782129526138306, -0.28689056634902954, -0.02351222187280655, -0.2224946767091751, -0.6145427227020264, -0.3284415006637573, -0.23217777907848358, 0.0934421718120575, 0.08927428722381592, 0.044960953295230865, -0.21556256711483002, 0.008354521356523037, 0.03142101317644119, -0.17217983305454254, -0.1872663050889969, -0.1010512113571167, -0.08459692448377609, -0.3158825635910034, -0.3358093798160553, -0.0008127720793709159, 0.6255096793174744, -0.0694938525557518, -0.10058903694152832, -0.1064097210764885, -0.4095796048641205, -0.5076856017112732, -0.2282649427652359, 0.13615188002586365, -0.22441649436950684, -0.318145751953125, 0.01708819717168808, -0.45441949367523193, 0.03560032695531845, -0.38840460777282715, 0.027887094765901566, 0.42600640654563904, -0.31313538551330566, -0.14107654988765717, 0.17949102818965912, -0.2942504584789276, -0.16662265360355377, -0.5167726278305054, -0.14809323847293854, -0.11052113026380539, 0.17146345973014832, -0.2105737030506134, -0.14836734533309937, -0.08710397779941559, 0.014715591445565224, -0.2895311117172241, -0.35780590772628784, -0.17525936663150787, 0.4654582142829895, -0.0349448136985302, -0.05074208229780197, -0.0033816543873399496, 0.10687919706106186, -0.41314995288848877, 0.02676340565085411, 0.33371463418006897, 0.4004887044429779, -0.2873215973377228, -0.16250742971897125, -0.23491255939006805, -0.1221236065030098, -0.14324212074279785, -0.25667449831962585, -0.04904274269938469, -0.013096216134727001, -0.26202264428138733, -0.2690410315990448, 0.49657195806503296, -0.22361236810684204, -0.18643318116664886, 0.23731732368469238, -0.3355777859687805, -0.06740938127040863, 0.18068161606788635, -0.2659361958503723, 0.11824405193328857, 0.03968995064496994, -0.3704856336116791, 0.12066896259784698, -0.2103719413280487, 0.005269820336252451, -0.286861389875412, -0.12885235249996185, 0.2396400421857834, -0.22918297350406647, 0.0014637497952207923, 0.20544438064098358, -0.08707555383443832, -0.3224799335002899, -0.220765620470047, -0.0648903101682663, -0.16174830496311188, 0.2768494784832001, -0.3594353199005127, -0.1261003315448761, -0.23292694985866547, 0.08843968063592911, 0.1028018444776535, 0.16082747280597687, -0.23006539046764374, -0.3676426112651825, -0.15405823290348053, -0.026368089020252228, -0.024782054126262665, -0.05520687252283096, -0.19677671790122986, -0.1750459223985672, -0.04840253293514252, -0.2496221661567688, 0.3134704828262329, -0.041046347469091415, -0.053727518767118454, 0.24883724749088287, -0.3146519660949707, 0.0491727814078331, 0.012099520303308964, -0.20169728994369507, -0.39943620562553406, -0.18044312298297882, -0.2250353991985321, -0.07337777316570282, 0.057019732892513275, -0.04520192742347717, -0.35758453607559204, -0.09028833359479904, -0.0005368694546632469, -0.15525196492671967, 0.0551646426320076, -0.23333214223384857, 0.04522872343659401, -0.08901491016149521, -0.3630845248699188, -0.34015941619873047, -0.1498093605041504, -0.11801528185606003, 0.11814918369054794, -0.41851329803466797, -0.4279351234436035, -0.1034330427646637, -0.21405960619449615, 0.47660788893699646, 0.2571631968021393, -0.20319293439388275, 0.10051945596933365, -0.2262897491455078, -0.16189566254615784, 0.1671096682548523, 0.12602265179157257, 0.2357196807861328, 0.512580931186676, -0.5589640736579895, -0.25800028443336487, 0.3024458587169647, -0.16784729063510895, -0.2676328122615814, -0.4560176432132721, -0.09102743864059448, -0.11097538471221924, -0.102625772356987, 0.34055182337760925, -0.5405901670455933, -0.28840959072113037, -0.07626229524612427, -0.2358076423406601, 0.2579776346683502, -0.0770709216594696, -0.3445775806903839, -0.2736998200416565, -0.4075256884098053, -0.5477306842803955, -0.09956742078065872, 0.07613677531480789, -0.3499484360218048, 0.30353131890296936, 0.022766387090086937, -0.013894597068428993, 0.2668670117855072, 0.019751347601413727, -0.11244097352027893, -0.0024488933850079775, -0.23118063807487488, -0.14949072897434235, 0.2576042413711548, -0.03145100548863411, -0.07640867680311203, -0.3695521056652069, -0.2485247403383255, 0.3284950256347656, -0.1543320119380951, -0.0775146484375, -0.04472629725933075, -0.35298293828964233, -0.04626164957880974, 0.6623384356498718, -0.06752464175224304, 0.023151734843850136, 0.0899631455540657, 0.017516901716589928, -0.022618846967816353, 0.05589621886610985, -0.09118636697530746, -0.21465665102005005, -0.24739067256450653, 0.0871601402759552, 0.2619859278202057, -0.13502204418182373, -0.00209827977232635, 0.1736326664686203, -0.3205799162387848, -0.2472909539937973, 0.27751779556274414, 0.580403745174408, -0.08156931400299072, 0.14110992848873138, 0.04376126453280449, -0.2670900821685791, 0.11167014390230179, 0.09421227127313614, -0.1320495754480362, -0.1064729169011116, -0.16652728617191315, -0.2545725107192993, 0.11656386405229568, -0.12657354772090912, -0.28208833932876587, -0.5653954744338989, 0.0520566925406456, 0.0851694718003273, -0.04037923738360405, -0.05475487560033798, 0.014877356588840485, 0.19246596097946167, -0.18191032111644745, 0.3006727397441864, -0.13984991610050201, 0.009779315441846848, -0.20575718581676483, 0.12978847324848175, -0.3504183292388916, 0.009562204591929913, -0.16624902188777924, 0.05847138166427612, 0.21823079884052277, 0.06453166157007217, -0.34365394711494446, -0.3381801247596741, 0.23715098202228546, 0.21896426379680634, -0.23303186893463135, -0.20397156476974487, -0.16570904850959778, -0.3843296766281128, -0.1289883553981781, -0.2252349704504013, -0.13726934790611267, 0.34607574343681335, 0.15231657028198242, 0.092666395008564, 0.1703280210494995, -0.13712938129901886, -0.18558312952518463, -0.17095793783664703, 0.045973408967256546, 0.006200634874403477, -0.11001211404800415, -0.13866889476776123, 0.2412092536687851, 0.3443906009197235, 0.031059039756655693, 0.2093174159526825, -0.2569422423839569, -0.08702262490987778, -0.0474107451736927, -0.1346384584903717, 0.25444385409355164, -0.1372525542974472, -0.011269254609942436, -0.19297006726264954, -0.26756948232650757, -0.29045727849006653, 0.17028509080410004, 0.052460893988609314, 0.038823649287223816, -0.05558585748076439, -0.07866787910461426, 0.3739311695098877, -0.46680203080177307, 0.10248074680566788, 0.23865802586078644, 0.07920388132333755, 0.06412546336650848, -0.2599007487297058, -0.0870715007185936, -0.16450491547584534, 0.11521106958389282, -0.2349354773759842, 0.05313846468925476, -0.3131425976753235, -0.0828932523727417, -0.06345212459564209, -0.043953415006399155, -0.15600837767124176, 0.3002765476703644, -0.2845238149166107, 0.055020771920681, 0.17754268646240234, 0.02041620761156082, -0.1085420772433281, 0.08243180066347122, -0.06968578696250916, 0.08361905068159103, -0.05880530923604965, 0.3440574109554291, -0.07880387455224991, -0.21437698602676392, -0.5288702249526978, -0.04821496084332466, 0.056090984493494034, -0.18004004657268524, 0.12095144391059875, -0.3894447684288025, -0.3574558198451996, -0.39204198122024536, -0.04335843771696091, -0.2529280483722687, 0.07991056144237518, -0.5437672138214111, -0.12676258385181427, 0.3752340078353882, 0.38710999488830566, -0.04004378989338875, -0.2295038104057312, -0.027778513729572296, 0.26858288049697876, -0.06608390063047409, 0.20666879415512085, 0.2215060442686081, 0.08669348806142807, 0.2385953813791275, 0.34990015625953674, 0.5121641159057617, 0.009347861632704735, 0.1465306580066681, 0.24737469851970673, -0.30146485567092896, 0.0921599268913269, 0.03650215268135071, 0.16508258879184723, 0.07854178547859192, 0.1306568831205368, 0.18489615619182587, 0.005275323521345854, -0.039671652019023895, -0.12136740982532501, -0.08410369604825974, 0.15893049538135529, -0.14498601853847504, 0.02626759745180607, -0.06393727660179138, 0.04902191832661629, 0.13569264113903046, 0.16354428231716156, -0.11775851249694824, 0.19539614021778107, -0.18516229093074799, 0.13782525062561035, 0.11066857725381851, 0.0735667422413826, -0.028427163138985634, 0.1448001265525818, 0.1881381869316101, 0.10356868803501129, -0.39910411834716797, -0.29929301142692566, 0.16535186767578125, 0.16928847134113312, -0.10302630811929703, -0.09401780366897583, -0.16675324738025665, -0.1476602554321289, -0.08496789634227753, -0.2463032603263855, -0.310535192489624, 0.20711608231067657, 0.1252504289150238, 0.3891475200653076, 0.20702217519283295, -0.10884897410869598, 0.16353952884674072, -0.08615674078464508, 0.11291410773992538, -0.07889186590909958, -0.20718979835510254, 0.15641117095947266, -0.3857727646827698, 0.07173090428113937, 0.37856459617614746, 0.0713178738951683, 0.01852286048233509, 0.08404898643493652, 0.23078316450119019, 0.15964868664741516, -0.2271527200937271, -0.11821839958429337, -0.327358216047287, -0.15599828958511353, -0.04249901324510574, 0.24840591847896576, -0.11939942091703415, -0.0983782410621643, 0.3507983386516571, -0.38473597168922424, 0.02219008095562458, -0.038150303065776825, -0.28427740931510925, -0.3458978831768036, 0.001519380952231586, 0.12016656994819641, 0.16286435723304749, 0.11215189844369888, 0.1204991489648819, -0.16519145667552948, 0.08245831727981567, -0.11128875613212585, -0.27699899673461914, -0.0779690146446228, 0.035389337688684464, 0.012477101758122444, 0.4441850483417511, -0.2875117063522339, -0.0431063175201416, -0.06309492886066437, -0.3026803135871887, -0.12670566141605377, -0.05282760411500931, 0.2409372478723526, 0.3760317265987396, -0.15556922554969788, -0.11303092539310455, -0.1377658247947693, 0.31025272607803345, 0.35855841636657715, -0.23127605020999908, -0.057934969663619995, 0.004590650554746389, -4.7083114623092115e-05, -0.08027546107769012, -0.2546803951263428, 0.0344395637512207, 0.13027936220169067, 0.1407293826341629, -0.07538329809904099, -0.17394517362117767, 0.36648261547088623, 0.04801194369792938, 0.23403045535087585, -0.2383449375629425, -0.11694535613059998, 0.2871584892272949, -0.3395628333091736, 0.03341652452945709, 0.007815578021109104, -0.12866359949111938, 0.171553835272789, 0.235432431101799, 0.6750158071517944, -0.2076561450958252, 0.24867498874664307, 0.003383868606761098, 0.0020409708376973867, -0.14047563076019287, -0.03168291226029396, 0.28884270787239075, 0.123771071434021, -0.012901033274829388, 0.22375644743442535, 0.18752367794513702, 0.12024352699518204, -0.0037902905605733395, -0.026948628947138786, -0.0773172527551651, -0.25981834530830383, -0.19098569452762604, 0.09582114219665527, -0.06051073968410492, -0.04218173772096634, -0.07756919413805008, 0.009212418459355831, 0.2981061339378357, -0.1319645345211029, -0.47778475284576416, 0.0006480926531367004, -0.22521258890628815, -0.0202554389834404, -0.17971171438694, 0.17691852152347565, -0.29180440306663513, 0.042712241411209106, -0.11077575385570526, -0.09951100498437881, 0.031430430710315704, -0.1737544685602188, 0.2709721624851227, -0.10065419226884842, 0.15368328988552094, 0.24572162330150604, -0.07264989614486694, -0.3360179662704468, 0.19760292768478394, 0.19272169470787048, -0.3120988607406616, 0.19702422618865967, -1.2429172784322873e-05, -0.18634305894374847, 0.10248443484306335, 0.060762397944927216, -0.02105189673602581, 0.043352559208869934, 0.1139506846666336, -0.1161152794957161, -0.030919218435883522, -0.12080781906843185, -0.15443168580532074, 0.034152306616306305, 0.1724671572446823, 0.1636205017566681, 0.21272355318069458, 0.20696046948432922, -0.2528083026409149, -0.1149296760559082, 0.3945447504520416, 0.11962136626243591, 0.0658361092209816, -0.17844869196414948, 0.10101983696222305, -0.3148329555988312, 0.4142385721206665, 0.06366857141256332, 0.10179205983877182, 0.22008642554283142, 0.02746562287211418, -0.17025846242904663, 0.3946748971939087, 0.1818298101425171, 0.03876325860619545, 0.20158320665359497, 0.4894409775733948, 0.09782713651657104, 0.13953165709972382, -0.21568313241004944, 0.5594409704208374, -0.3128034770488739, -0.022325988858938217, -0.014111945405602455, 0.12628374993801117, -0.08897629380226135, 0.3117372691631317, -0.3928244411945343, -0.07835248857736588, -0.1788617968559265, -0.29344743490219116, 0.042241353541612625, 0.006705728359520435, -0.08214955776929855, 0.12000124901533127, 0.45589685440063477, -0.0829542875289917, 0.13357748091220856, -0.05355854332447052, -0.07673729956150055, 0.15106163918972015, 0.30951279401779175, 0.014486338943243027, -0.4886089861392975, 0.1807357221841812, -0.18467053771018982, 0.14209607243537903, 0.5688550472259521, -0.12144497781991959, -0.19225221872329712, -0.012114367447793484, -0.1275583654642105, -0.22525501251220703, -0.12953993678092957, 0.018923740833997726, -0.029003748670220375, -0.1692586988210678, 0.38832801580429077, 0.21871086955070496, -0.11404621601104736, 0.15502801537513733, -0.13513077795505524, -0.3113378882408142, -0.16474464535713196, 0.19383305311203003, -0.15348781645298004, -0.30608850717544556, -0.012301962822675705, -0.3232949674129486, 0.0975390076637268, -0.03084494173526764, -0.1071132943034172, 0.6795222163200378, -0.12324412167072296, -0.060258325189352036, 0.2621717154979706, -0.12022644281387329, -0.02296886406838894, 0.03619205579161644, 0.011585366912186146, -0.14347884058952332, 0.14538785815238953, -0.24317137897014618, 0.10569644719362259, 0.394788920879364, -0.2311607003211975, -0.40375062823295593, 0.016589317470788956, -0.18221087753772736, 0.21423250436782837, -0.10904121398925781, 0.025772685185074806, 0.03301217406988144, 0.12305178493261337, -0.04276375100016594, 0.048648443073034286, -0.025948254391551018, -0.2866119146347046, -0.3000202178955078, 0.23716123402118683, 0.05681716650724411, 0.15931347012519836, -0.2097620666027069, -0.02668263204395771, -0.2616916000843048, -0.2684423625469208, -0.16149136424064636, 0.16588695347309113, -0.0719744935631752, -0.0008606349001638591, 0.07452009618282318, 0.098502017557621, -0.05962285399436951, -0.15250453352928162, -0.04913247004151344, -0.057089515030384064, -0.1088332086801529, -0.28368526697158813, 0.06596271693706512, -0.25890275835990906, 0.0828428789973259, -0.0246073380112648, -0.1252584457397461, -0.1770821511745453, -0.11318474262952805, 0.22155147790908813, -0.32400256395339966, 0.00727829709649086, 0.17147812247276306, 0.16137568652629852, -0.03006889671087265, 0.038237083703279495, -0.004765362478792667, -0.31681784987449646, -0.017093995586037636, -0.1493866741657257, -0.4987230896949768, 0.12450442463159561, 0.04196247458457947, -0.2788824439048767, -0.1555626094341278, 0.0995018258690834, -0.10540684312582016, 0.0722769945859909, -0.20505523681640625, 0.007406650576740503, 0.1328115314245224, 0.2975602149963379, 0.056749265640974045, -0.04041143134236336, 0.5152500867843628, -0.18227554857730865, -0.13669942319393158, -0.13680081069469452, -0.4438895881175995, -0.0936407670378685, -0.026187365874648094, -0.03669436648488045, 0.16297629475593567, -0.04460667446255684, 0.03976026177406311, -0.025837330147624016, -0.0884566530585289, 0.1522364616394043, 0.061178795993328094, -0.18845394253730774, -0.1127735897898674, -0.03358481451869011, -0.11362442374229431, -0.5058112740516663, 0.09294590353965759, -0.18261925876140594, -0.022777004167437553, 0.02826399728655815, -0.01755603961646557, -0.21169912815093994, -0.1765441596508026, 0.14239707589149475, 0.1309269815683365, 0.09460562467575073, -0.26193687319755554, 0.04567578434944153, 0.11704176664352417, 0.4615779221057892, -0.002539282198995352, 0.10626433044672012, 0.15655948221683502, 0.00036121884477324784, -0.14015604555606842, -0.1536766141653061, -0.09798014163970947, 0.04883619025349617, 0.0797182247042656, 0.004987557418644428, 0.06826328486204147, 0.24519047141075134, -0.015432516112923622, -0.07295110821723938, 0.26977378129959106, -0.008795753121376038, -0.43883204460144043, -0.489227831363678, 0.30218029022216797, 0.1914844661951065, 0.16463319957256317, 0.2464660257101059, 0.009895682334899902, 0.011003333143889904, 0.1821073591709137, -0.121885746717453, -0.16312366724014282, -0.15869811177253723, -0.20886573195457458, 0.21728552877902985, -0.1350230723619461, 0.3851223289966583, -0.16199035942554474, 0.12983004748821259, 0.12516401708126068, 0.15819001197814941, 0.022599296644330025, 0.0576481893658638, -0.0410541296005249, 0.32949987053871155, 0.09495435655117035, -0.36032092571258545, -0.11618708819150925, 0.11133292317390442, 0.20637747645378113, 0.15929576754570007, 0.14175966382026672, 0.09930524975061417, 0.015641603618860245, -0.47138071060180664, -0.10532421618700027, 0.1270076036453247, -0.2432834357023239, -0.05390181392431259, -0.2727794945240021, 0.09569257497787476, 0.2615889608860016, 0.15654493868350983, 0.21117833256721497, -0.13012883067131042, -0.09427247941493988, 0.002183233154937625, -0.09125394374132156, 0.21163196861743927, -0.14500436186790466, 0.010758128017187119, 0.21012766659259796, -0.02746051549911499, -0.1756710559129715, 0.1538231521844864, 0.28954747319221497, 0.04843192547559738, -0.20011599361896515, -0.09129427373409271, -0.09168310463428497, 0.13139620423316956, 0.03811036795377731, 0.19850793480873108, 0.2777765095233917, 0.23547042906284332, -0.10635360330343246, -0.11519412696361542, -0.1208944246172905, 0.1347927302122116, -0.13775482773780823, 0.041410062462091446, -0.29137513041496277, 0.09179913252592087, -0.09648251533508301, -0.10561065375804901, 0.013621479272842407, 0.25928807258605957, 0.17363430559635162, -0.13552843034267426, 0.11499332636594772, 0.03987588733434677, 0.16145963966846466, 0.003481586929410696, 0.22650311887264252, -0.07476057857275009, -0.13185012340545654, 0.033866286277770996, -0.1305733323097229, -0.061058737337589264, 0.06394129246473312, 0.024490943178534508, -0.02172066643834114, -0.02965015359222889, 0.11849872767925262, 0.32412418723106384, 0.3391115069389343, 0.16906939446926117, 0.01337528321892023, 0.039516206830739975, -0.21563075482845306, 0.3815601170063019, -0.036305733025074005, -0.03956560418009758, 0.023301804438233376, -0.04759128764271736, 0.22963584959506989, 0.020569995045661926, -0.07781969755887985, -0.306919664144516, -0.05597775802016258, -0.23621079325675964, 0.3529980480670929, 0.13619865477085114, -0.08482032269239426, 0.07364042103290558, -0.03048449382185936, 0.12464354187250137, -0.013925163075327873, 0.03763674199581146, 0.0005325314123183489, -0.11279510706663132, 0.05012354999780655, 0.07905643433332443, 0.07302657514810562, -0.12218690663576126, -0.365915983915329, -0.07115810364484787, -0.06167230382561684, 0.18647855520248413, 0.15008065104484558, 0.3413083553314209, -0.19401073455810547, 0.30713018774986267, -0.16167433559894562, 0.3876280188560486, 0.03228013962507248, 0.08464539796113968, -0.20214350521564484, 0.16007967293262482, -0.19313718378543854, 0.05101725831627846, 0.37627294659614563, 0.040079932659864426, 0.1676948517560959, 0.26483187079429626, -0.310874879360199, 0.16560626029968262, -0.0694105476140976, 0.013753769919276237, 0.218533456325531, -0.07517004758119583, -0.14729295670986176, -0.03645315766334534, 0.11646895855665207, 0.04190315678715706, -0.25720420479774475, -0.26092055439949036, -0.07329986244440079, -0.021356690675020218, -0.1772833615541458, -0.2886965572834015, -0.10631772130727768, 0.2046036273241043, -0.2591916620731354, 0.5167078375816345, -0.3720373213291168, -0.26252689957618713, -0.092686727643013, -0.1430305540561676, 0.0781475231051445, 0.07110437005758286, 0.37043604254722595, 0.33514028787612915, -0.0172318946570158, -0.11596231907606125, 0.04533497989177704, 0.10475978255271912, -0.12519074976444244, 0.13757586479187012, 0.18031048774719238, 0.07914485037326813, -0.02038642205297947, -0.10345508903265, -0.2036171555519104, -0.1091790422797203, 0.12095216661691666, -0.11435094475746155, 0.17184823751449585, 0.05180178955197334, 0.09517384320497513, 0.3147287964820862, 0.31200775504112244, 0.07934567332267761, -0.0008261938346549869, 0.03950860723853111, 0.26642152667045593, -0.01930183731019497, -0.11295123398303986, -0.16731709241867065, 0.02847929485142231, -0.008164145052433014, -0.02952558547258377, -0.1022537350654602, 0.2379615157842636, -0.18593625724315643, 0.17528052628040314, 0.12337192893028259, 0.1875562071800232, -0.27618786692619324, 0.033955834805965424, 0.10565312206745148, 0.04008878022432327, 0.06265266239643097, -0.1865701973438263, 0.10647094994783401, 0.2052006870508194, 0.17696982622146606, -0.033234186470508575, -0.11100930720567703, 0.06142749637365341, -0.10419251769781113, -0.4270409345626831, 0.07946634292602539, 0.3361113965511322, -0.14617736637592316, -0.1121908500790596, -0.3260013163089752, 0.33257731795310974, 0.10829604417085648, -0.019141951575875282, 0.17280466854572296, -0.09055652469396591, 0.03171152248978615, -0.0525527149438858, -0.07125657796859741, 0.19926561415195465, 0.18365441262722015, 0.08738469332456589, -0.03403780236840248, 0.2849641442298889, 0.008462093770503998, -0.0686006173491478, -0.16501255333423615, 0.10478740930557251, -0.3338509798049927, 0.2566268742084503, 0.0013469421537593007, -0.13521835207939148, 0.0071938177570700645, 0.25188493728637695, -0.160487100481987, 0.16128219664096832, 0.2765131890773773, -0.017360873520374298, 0.22243013978004456, -0.29473331570625305, 0.19231435656547546, 0.10081209242343903, 0.21141721308231354, -0.35231050848960876, 0.11062648147344589, 0.053225982934236526, -0.12106454372406006, -0.01688958890736103, 0.03336510807275772, -0.0449226088821888, 0.045591454952955246, 0.2265891581773758, 0.14235348999500275, -0.3955855965614319, -0.003227650886401534, 0.17384162545204163, -0.27818363904953003, 0.2713286578655243, 0.024190492928028107, -0.23295573890209198, 0.25213539600372314, -0.014903275296092033, -0.035388607531785965, -0.17943145334720612, 0.4432533383369446, 0.06272491812705994, 0.08793026208877563, 0.2713131904602051, 0.0009119801688939333, 0.15456563234329224, 0.04930215701460838, 0.1267099529504776, 0.07204634696245193, 0.2191508412361145, -0.019905459135770798, 0.11200731992721558, -0.00020899575611110777, -0.07623293995857239, -0.35249122977256775, -0.023083878681063652, 0.0321798212826252, 0.09063490480184555, -0.18862952291965485, 0.16523712873458862, 0.1037612184882164, -0.010841812007129192, 0.14105981588363647, 0.13610149919986725, 0.12595753371715546, -0.3959442377090454, 0.25352242588996887, 0.2516195774078369, -0.21423819661140442, -0.20373527705669403, -0.2951074242591858, -0.023310063406825066, -0.08050110191106796, -0.05698683112859726, 0.23346249759197235, 0.6358658075332642, 0.019458243623375893, -0.39494937658309937, 0.09179695695638657, 0.3716895580291748, -0.2739529013633728, -0.3240106403827667, -0.03757062554359436, 0.040084511041641235, 0.4323256313800812, 0.1773253232240677, 0.05099697783589363, -0.013918153941631317, -0.22594588994979858, 0.07446993142366409, -0.027757292613387108, -0.24223117530345917, 0.07882622629404068, -0.25876110792160034, 0.09381971508264542, -0.13343676924705505, 0.17815537750720978, 0.17777970433235168, -0.3067653775215149, -0.16189545392990112, 0.1378510445356369, 0.10569486021995544, -0.04758394509553909, 0.16709548234939575, -0.05300604924559593, 0.25497421622276306, -0.200638085603714, 0.30013182759284973, 0.18673397600650787, -0.0016968921991065145, -0.235408753156662, 0.07888542115688324, 0.09421974420547485, -0.10437767952680588, -0.005095452535897493, -0.05182942748069763, 0.11754190921783447, 0.11361844837665558, -0.1178659051656723, -0.02324623614549637, -0.05452142283320427, 0.09321214258670807, -0.15023401379585266, -0.16000248491764069, -0.16531574726104736, 0.20847906172275543, -0.019285358488559723, -0.20805154740810394, -0.11417771875858307, -0.6670738458633423, 0.2697972357273102, 0.03457298502326012, -0.16001705825328827, 0.1514636129140854, 0.4524340033531189, -0.24437777698040009, -0.19873008131980896, 0.39561915397644043, -0.06721494346857071, 0.03295445814728737, -0.1680033802986145, 0.1690947413444519, -0.23324795067310333, -0.10850788652896881, 0.004190400242805481, 0.010297882370650768, -0.05938805267214775, -0.0595753937959671, 0.2817796766757965, -0.2821723520755768, -0.007516409270465374, 0.38836321234703064, -0.2970801293849945, -0.07927516847848892, 0.04496103525161743, -0.24053418636322021, -0.11921290308237076, -0.1669062376022339, 0.13351571559906006, -0.023243268951773643, -0.3323546051979065, 0.26441553235054016, 0.23714835941791534, -0.1449616402387619, -0.14985397458076477, -0.20037910342216492, 0.3495168387889862, 0.2050391584634781, 0.04254930093884468, -0.21899032592773438, -0.07918954640626907, -0.014390259049832821, 0.12801234424114227, 0.27869048714637756, -0.33480092883110046, -0.021937496960163116, 0.4042811691761017, 0.14783768355846405, 0.09359237551689148, -0.06178606301546097, 0.13090898096561432, 0.06981810182332993, 0.132765531539917, 0.13222245872020721, -0.049536582082509995, -0.18912988901138306, 0.04012219235301018, 0.131022647023201, -0.13037873804569244, -0.07606881111860275, -0.005796114448457956, -0.049590494483709335, 0.19698289036750793, 0.136505126953125, -0.004938995465636253, -0.019750408828258514, 0.17368945479393005, -0.07746796309947968, 0.03594452887773514, 0.2295903116464615, 0.16159094870090485, -0.5043202638626099, 0.07618264853954315, 0.1162082627415657, -0.06238100677728653, 0.3323797881603241, -0.07470112293958664, -0.10084322094917297, -0.016101254150271416, -0.06160538271069527, -0.013350422494113445, -0.07088454812765121, 0.13008935749530792, -0.14576931297779083, -0.26238393783569336, 0.10909896343946457, -0.00984195712953806, 0.4236793518066406, -0.2746221721172333, -0.15576843917369843, 0.07911593466997147, 0.04058096557855606, 0.06069926917552948, 0.010668616741895676, -0.019655412063002586, -0.21293409168720245, -0.2163526564836502, 0.12477192282676697, 0.01822511851787567, 0.24023887515068054, 0.06155572459101677, -0.2512121796607971, 0.3418971002101898, 0.15773974359035492, 0.17538617551326752, -0.006558042950928211, 0.034282635897397995, -0.08391915261745453, 0.1859971284866333, 0.21265412867069244, 0.14821957051753998, 0.11003678292036057, -0.32292604446411133, 0.3346458077430725, -0.28126686811447144, -0.03062785230576992, -0.06220445781946182, 0.042297687381505966, -0.17628361284732819, 0.018886420875787735, -0.10089430212974548, 0.005934880580753088, 0.16868716478347778, 0.23653212189674377, 0.2597864270210266, -0.007172789890319109, -0.11186417192220688, -0.24851195514202118, -0.17189674079418182, -0.2477010190486908, 0.146396204829216, -0.047849129885435104, 0.4983243942260742, -0.21633492410182953, 0.02954317256808281, 0.24314819276332855, 0.14176979660987854, 0.1845712959766388, 0.09106002002954483, 0.10957429558038712, -0.05480419099330902, -0.007242175284773111, -0.03539358824491501, 0.09193486720323563, 0.12157420814037323, -0.13221006095409393, -0.19778303802013397, 0.100596584379673, 0.023037318140268326, -0.14032107591629028, -0.28538042306900024, 0.26501718163490295, -0.2561163902282715, -0.2810872495174408, -0.14381828904151917, 0.07363823801279068, 0.05536023527383804, -0.0965009331703186, -0.03428685665130615, 0.296631395816803, -0.12011975049972534, -0.09797720611095428, 0.639039933681488, 0.12160186469554901, 0.4061860740184784, 0.2674309015274048, -0.03691457211971283, 0.4371851980686188, -0.1377558708190918, 0.43088531494140625, 0.0968189388513565, 0.20543347299098969, -0.08829324692487717, -0.2520018219947815, -0.05408494547009468, -0.11601576209068298, -0.02118314988911152, -0.07121694833040237, -0.09069859236478806, -0.18125514686107635, -0.08378124237060547, -0.08536526560783386, 0.09135515987873077, 0.01322726160287857, 0.09304064512252808, 0.23750342428684235, -0.34305888414382935, -0.07674521952867508, 0.2892293334007263, 0.3205324411392212, -0.16416461765766144, -0.0524284653365612, 0.04735260084271431, 0.11964055150747299, 0.21372103691101074, -0.1320979744195938, 0.04353196546435356, 0.34387123584747314, -0.23912368714809418, -0.24102696776390076, -0.1338946372270584, -0.10704637318849564, -0.21065858006477356, -0.17386411130428314, -0.06122000887989998, -0.07745316624641418, -0.03499382734298706, 0.08475272357463837, 0.18018625676631927, -0.3687748610973358, 0.10364647954702377, -0.027723237872123718, 0.14718274772167206, -0.26408785581588745, -0.04487522318959236, -0.2702990472316742, 0.280966579914093, -0.3591945767402649, -0.31158247590065, 0.22974923253059387, 0.21828606724739075, -0.06789033114910126, 0.07904708385467529, 0.1801125854253769, 0.03467734158039093, 0.09174428135156631, 0.3197823464870453, 0.21389666199684143, -0.4700510799884796, -0.006770632229745388, -0.25416016578674316, -0.07299597561359406, -0.2657797932624817, -0.12120123207569122, 0.1733037680387497, -0.04629700258374214, -0.1748448610305786, -0.13183242082595825, -0.05907456949353218, -0.10312524437904358, 0.009652500040829182, 0.2549929618835449, 0.10806821286678314, -0.4217376410961151, 0.43450504541397095, 0.07565303146839142, 0.006166104692965746, 0.03408987820148468, -0.11869164556264877, -0.12728068232536316, -0.06212690845131874, 0.1515541523694992, -0.01893078163266182, 0.20271332561969757, 0.2002715766429901, -0.17555418610572815, -0.09793157130479813, -0.10659287869930267, -0.23892901837825775, 0.042177483439445496, 0.013625643216073513, -0.050937436521053314, -0.24944759905338287, -0.15493550896644592, 0.38972392678260803, 0.12800894677639008, -0.027462899684906006, 0.08557499945163727, 0.013978966511785984, 0.2510592043399811, 0.31056100130081177, 0.2867332696914673, -0.012855133973062038, -0.026613811030983925, 0.24856123328208923, -0.018577003851532936, -0.11289455741643906, 0.2245320975780487, -0.02510877326130867, 0.2600829601287842, -0.02485741674900055, 0.10663396120071411, 0.003429548814892769, -0.03015001304447651, 0.27966243028640747, 0.1651000678539276, 0.015445132739841938, -0.006777915172278881, -0.250348299741745, -0.06621658802032471, 0.09259333461523056, -0.28582531213760376, -0.007643376011401415, -0.06807807832956314, -0.14661240577697754, -0.11450479179620743, -0.15603888034820557, 0.14098197221755981, -0.05506233498454094, 0.2505755126476288, -0.01885155402123928, 0.024710943922400475, 0.03970028832554817, -0.09032491594552994, 0.13604845106601715, 0.36879202723503113, -0.3175417184829712, -0.2659083604812622, 0.22847053408622742, -0.1259903758764267, 0.05608562380075455, 0.03779875487089157, 0.04620014503598213, 0.06265877187252045, 0.13790827989578247, 0.11409661918878555, -0.12296023964881897, -0.10667440295219421, -0.3559158444404602, -0.06429069489240646, -0.08072725683450699, -0.3163050413131714, -0.003747494425624609, -0.012223082594573498, -0.01766875758767128, -0.051657721400260925, 0.1666756123304367, -0.2511287033557892, 0.11503687500953674, 0.14624442160129547, -0.06879914551973343, -0.11858005821704865, -0.13094688951969147, -0.4305388331413269, -0.01106554176658392, -0.07857432216405869, 0.07676831632852554, 0.2465646117925644, -0.2296789288520813, 0.08989894390106201, -0.12468885630369186, 0.07596343755722046, -0.3983185887336731, -0.015582756139338017, 0.20802566409111023, -0.26879316568374634, 0.16112250089645386, 0.19876201450824738, 0.06967063993215561, 0.0649576187133789, 0.342345267534256, 0.1229056641459465, -0.08301553875207901, -0.07016861438751221, 0.37867051362991333, -0.10658137500286102, 0.24259358644485474, 0.027661141008138657, -0.29338306188583374, -0.06344877928495407, 0.06423855572938919, -0.13789597153663635, -0.23760546743869781, -0.11095184832811356, 0.1210629791021347, 0.1831231713294983, 0.2233496904373169, -0.29485511779785156, -0.0946199968457222, 0.1504717469215393, -0.16573798656463623, 0.3931785821914673, -0.15357479453086853, -0.3016524016857147, -0.051606811583042145, -0.278129518032074, -0.1258987933397293, -0.08587339520454407, 0.03197989985346794, -0.31891554594039917, -0.14098896086215973, -0.06280932575464249, 0.020081907510757446, -0.16227179765701294, 0.12816523015499115, -0.15469594299793243, -0.3388179838657379, -0.26214227080345154, 0.20931825041770935, 0.07136200368404388, 0.0876363068819046, 0.05784659460186958, -0.05366125702857971, -0.2282489538192749, -0.07847771793603897, -0.06586208194494247, 0.04325263947248459, -0.3327421545982361, 0.28414759039878845, 0.13590389490127563, -0.09204611927270889, 0.149074524641037, 0.1852141171693802, -0.24018044769763947, 0.24176864326000214, -0.24655374884605408, 0.5317663550376892, -0.09795619547367096, 0.0896175354719162, 0.1914600282907486, 0.09755854308605194, 0.2594917416572571, -0.1649957150220871, 0.42418724298477173, -0.07329012453556061, 0.007339010015130043, -0.09945684671401978, 0.05270152539014816, -0.10175689309835434, -0.05495477095246315, -0.1011478379368782, 0.014241812750697136, 0.39017245173454285, -0.05666494742035866, 0.09544580429792404, 0.4383413791656494, -0.014982098713517189, -0.360919326543808, -0.16499821841716766, -0.10617654025554657, -0.31339892745018005, -0.31461071968078613, 0.20705245435237885, -0.07683920115232468, -0.053842734545469284, 0.08915301412343979, 0.29491961002349854, 0.6350038051605225, 0.2259603887796402, 0.3095662295818329, 0.16793619096279144, 0.12976519763469696, 0.01426683459430933, -0.15263347327709198, 0.10845311731100082, 0.03933306038379669, 0.02803301066160202, 0.05495281144976616, 0.030037062242627144, -0.2724948823451996, -0.4431699514389038, 0.341617614030838, 0.28007054328918457, 0.08386272192001343, 0.2580782473087311, 0.07782252132892609, -0.21684880554676056, -0.0005563266458921134, -0.11221450567245483, -0.021366357803344727, -0.10412289947271347, 0.12500827014446259, 0.1333911120891571, 0.21542347967624664, 0.2918938100337982, 0.15678535401821136, -0.08363067358732224, 0.13394449651241302, 0.31784310936927795, 0.3843017816543579, -0.16999749839305878, -0.18981312215328217, 0.12813054025173187, -0.192709818482399, -0.054181840270757675, -0.02682107873260975, 0.11961715668439865, -0.2204805165529251, 0.07776738703250885, 0.11075419187545776, 0.25247299671173096, 0.06809217482805252, -0.17590901255607605, -0.06447518616914749, 0.21047979593276978, -0.04753632843494415, 0.05457102507352829, 0.04623902216553688, 0.11353692412376404, -0.13278476893901825, -0.2884645164012909, 0.04967989772558212, 0.08271083980798721, 0.20990602672100067, -0.03744134679436684, 0.15883196890354156, 0.22269542515277863, 0.08127322793006897, -0.14888633787631989, -0.09136103838682175, 0.06755571067333221, 0.10902395099401474, -0.2729766368865967, -0.11029376834630966, 0.11217924952507019, -0.22010071575641632, 0.11524281650781631, 0.565212607383728, -0.06793300062417984, 0.30051809549331665, -0.02822885848581791, -0.0631636455655098, -0.16099092364311218, -0.09586545079946518, -0.03221558406949043, 0.023252131417393684, 0.2537058889865875, 0.11985687911510468, 0.2669350802898407, 0.10982826352119446, 0.11082585155963898, -0.06186668202280998, 0.1401287019252777, 0.0759543776512146, -0.09103796631097794, 0.1695907860994339, 0.1044636219739914, -0.0900961309671402, -0.42156508564949036, -0.10809066146612167, 0.24892379343509674, 0.23209860920906067, -0.378730446100235, -0.05177126079797745, -0.02593320794403553, 0.04396775737404823, -0.046129804104566574, 0.08914121985435486, 0.34873491525650024, -0.1371273249387741, 0.1093619093298912, -0.3140239417552948, 0.05152813345193863, 0.09658589959144592, 0.38600364327430725, 0.251312255859375, -0.2630998492240906, -0.1326482594013214, 0.014324614778161049, -0.03433337062597275, -0.08430974185466766, 0.10450780391693115, 0.1434391438961029, 0.34308144450187683, 0.2643681764602661, 0.17892515659332275, -0.1291884183883667, -0.27442657947540283, 0.007584252394735813, 0.22184456884860992, 0.1787601113319397, -0.0298247542232275, -0.16449443995952606, -0.17228861153125763, 0.0977972149848938, -0.11770402640104294, 0.12536537647247314, -0.06030542403459549, 0.11639241874217987, 0.04037310928106308, -0.04079487919807434, -0.08013491332530975, 0.03391348570585251, -0.15650859475135803, 0.3412894904613495, 0.1457202434539795, -0.16717377305030823, 0.3081214129924774, 0.06337898969650269, 0.12699365615844727, -0.14530323445796967, -0.11012541502714157, 0.18331021070480347, 0.051879800856113434, 0.21335935592651367, -0.04935115948319435, -0.16894005239009857, 0.11012017726898193, 0.17311082780361176, 0.29548680782318115, 0.1525433212518692, -0.038604751229286194, 0.04909077659249306, 0.010139113292098045, 0.03877018764615059, 0.1936347633600235, 0.05677654221653938, -0.12165053188800812, 0.16979770362377167, 0.25255078077316284, -0.09155263751745224, 0.1909719705581665, 0.023594947531819344, 0.20724916458129883, 0.3555811047554016, -0.39288532733917236, 0.060564812272787094, -0.17831505835056305, -0.019284160807728767, -0.08578179776668549, -0.16899646818637848, -0.010993926785886288, -0.06943728029727936, 0.12144695222377777, 0.10644954442977905, -0.05913303419947624, -0.06763577461242676, 0.22916889190673828, 0.0013828582596033812, -0.2076895385980606, 0.12246328592300415, -0.13703057169914246, -0.11919901520013809, 0.11679242551326752, -0.05580572411417961, -0.03849005699157715, 0.1933414787054062, -0.12554650008678436, -0.09370630979537964, -0.09978264570236206, -0.07065615057945251, 0.3205324709415436, -0.34930118918418884, -0.030045725405216217, 0.08893312513828278, 0.10582086443901062, -0.057845715433359146, 0.16411283612251282, 0.08010472357273102, 0.1570287048816681, 0.35194826126098633, -0.030304670333862305, 0.014462439343333244, -0.029750054702162743, 0.29541444778442383, 0.10172438621520996, -0.19023096561431885, 0.18170276284217834, 0.019404582679271698, 0.19701933860778809, -0.08663322776556015, -0.057558201253414154, -0.4805976152420044, 0.037320610135793686, 0.032214369624853134, -0.009483518078923225, 0.3533126413822174, -0.050977546721696854, 0.16663195192813873, 0.09219363331794739, 0.07484777271747589, -0.23058748245239258, 0.2377762496471405, -0.11334096640348434, -0.1563492715358734, 0.4949330985546112, 0.03919577598571777, 0.18176975846290588, -0.0905553475022316, -0.11135513335466385, -0.0442521832883358, 0.25188297033309937, 0.13178271055221558, -0.06917505711317062, -0.2447923719882965, -0.2632260024547577, -0.08840527385473251, 0.1275576800107956, -0.21143801510334015, -0.1581357717514038, 0.3497816324234009, 0.17573459446430206, -0.1325797140598297, -0.0483551025390625, 0.09725422412157059, -0.12299531698226929, 0.044674381613731384, -0.09213849157094955, 0.43157467246055603, 0.23060746490955353, 0.05046594515442848, -0.11704952269792557, -0.10730171203613281, -0.09932709485292435, 0.31745460629463196, 0.11713022738695145, 0.3414798080921173, -0.17642942070960999, 0.014468910172581673, 0.2727162837982178, -0.026298774406313896, -0.023416155949234962, 0.08497851341962814, -0.0044555957429111, -0.11993113160133362, 0.0528445690870285, 0.269880086183548, 0.21439534425735474, 0.0030846481677144766, 0.19200196862220764, -0.1319287270307541, 0.15192197263240814, -0.0922974944114685, 0.03034309856593609, -0.11389249563217163, -0.0620112344622612, 0.20524132251739502, -0.11916981637477875, 0.08711350709199905, -0.04965290054678917, -0.09581901133060455, -0.040511712431907654, 0.08080243319272995, 0.21463879942893982, -0.03443625569343567, -0.2768562436103821, -0.17505469918251038, 0.1168464794754982, 0.1590520143508911, 0.30883902311325073, -0.06645358353853226, -0.11938953399658203, -0.20318666100502014, -0.07614443451166153, 0.30854830145835876, -0.16901744902133942, -0.035863056778907776, -0.051996294409036636, 0.138060063123703, 0.31859278678894043, -0.0717729926109314, 0.4609239995479584, 0.28264835476875305, 0.2752143442630768, 0.12007426470518112, 0.20358102023601532, -0.019745217636227608, -0.3572636842727661, 0.057684559375047684, 0.27826204895973206, 0.03073432482779026, 0.3461137115955353, -0.024597974494099617, -0.12036412954330444, -0.03236564248800278, 0.07074850052595139, 0.4218786060810089, 0.23942749202251434, -0.23642323911190033, -0.0787888765335083, -0.14522945880889893, 0.1485644429922104, -0.0990423932671547, 0.17813719809055328, 0.11730127781629562, -0.33279716968536377, 0.02194978855550289, 0.34215086698532104, 0.3129965364933014, -0.19260412454605103, -0.2807939946651459, -0.05703487992286682, 0.028835033997893333, -0.18546272814273834, -0.1780044138431549, 0.4666959047317505, 0.10816618800163269, 0.2972383201122284, -0.15718330442905426, 0.13232983648777008, -0.16521072387695312, 0.2864493727684021, 0.27914199233055115, 0.3392723500728607, -0.15691614151000977, 0.1785190850496292, -0.030978724360466003, 0.16928745806217194, -0.10819879919290543, 0.11088820546865463, -0.04455084353685379, 0.2773955464363098, 0.010030410252511501, 0.2688557505607605, 0.031225260347127914, -0.1237163320183754, 0.16855931282043457, 0.14218713343143463, 0.03609335049986839, 0.00898004975169897, 0.11157277971506119, 0.25501856207847595, 0.0984954759478569, 0.3336399793624878, 0.2228618562221527, 0.020865557715296745, -0.0685604065656662, 0.13629385828971863, 0.12387068569660187, 0.2003447562456131, 0.13128024339675903, 0.16638818383216858, 0.004248861689120531, -0.056163426488637924, -0.31421294808387756, -0.10776279866695404, -0.186579167842865, -0.054877255111932755, 0.12046029418706894, 0.20866192877292633, -0.3155229389667511, -0.04827645793557167, -0.10693369805812836, -0.07973716408014297, 0.004012943711131811, -0.2461576610803604, 0.17979562282562256, 0.1387939304113388, 0.08072090893983841, -0.06838921457529068, 0.07053632289171219, 0.1742105931043625, -0.30319908261299133, 0.07088983803987503, 0.08217959105968475, 0.29081517457962036, 0.23621410131454468, -0.1354905664920807, -0.18209590017795563, 0.122854083776474, -0.05998550355434418, -0.13626468181610107, 0.18849807977676392, -0.03190075606107712, 0.031425315886735916, 0.012899108231067657, 0.1497497856616974, 0.055759258568286896, 0.42072638869285583, 0.1292172223329544, 0.24006877839565277, 0.10719401389360428, -0.032620467245578766, 0.2899300456047058, 0.014187939465045929, -0.2948288917541504, 0.0182203222066164, -0.10200770199298859, 0.24616260826587677, -0.08718669414520264, 0.10812041908502579, -0.16289696097373962, 0.2077866941690445, -0.14145246148109436, 0.41495922207832336, 0.23739813268184662, 0.40946897864341736, 0.20598308742046356, 0.03517662733793259, -0.08970148861408234, -0.3496006429195404, 0.22167827188968658, 0.011081800796091557, -0.07633259892463684, -0.3666705787181854, -0.03850029036402702, 0.039288416504859924, -0.12558113038539886, 0.3538869023323059, 0.0333036407828331, -0.3090854287147522, -0.125340074300766, 0.10068424791097641, 0.0848441794514656, -0.18151821196079254, 0.07744847238063812, -0.2049015462398529], "yaxis": "y"}],
                        {"legend": {"itemsizing": "constant", "tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "x0"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "x1"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('8c451939-f6de-4f63-8af9-ade796be4434');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
weights
```




    array([[ 1.6684402e-02, -5.2643596e-04],
           [-8.7498590e-02, -4.0375008e-03],
           [ 1.9110104e+00,  7.6016746e-02],
           ...,
           [-2.6335064e-01, -6.2618703e-02],
           [-2.7115735e-01, -7.5890809e-02],
           [-9.9839546e-02, -4.5861937e-02]], dtype=float32)




```python

```
