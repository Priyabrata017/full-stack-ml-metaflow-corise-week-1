from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# TODO move your labeling function from earlier in the notebook here
labeling_function = lambda row: 0 if row['rating'] < 4 else 1

class GoodFirstModelNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline)

    # def RNN():
    #     inputs = Input(name='inputs',shape=[max_len])
    #     layer = Embedding(max_words,50,input_length=max_len)(inputs)
    #     layer = LSTM(64)(layer)
    #     layer = Dense(256,name='FC1')(layer)
    #     layer = Activation('relu')(layer)
    #     layer = Dropout(0.5)(layer)
    #     layer = Dense(1,name='out_layer')(layer)
    #     layer = Activation('sigmoid')(layer)
    #     model = Model(inputs=inputs,outputs=layer)
    # return model

    @step
    def baseline(self):
        "Compute the baseline"
        
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        max_words = 1000
        max_len = 150
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(self.traindf['review'])
        sequences = tok.texts_to_sequences(self.traindf['review'])
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

        from sklearn.metrics import accuracy_score, roc_auc_score
        inputs = Input(name='inputs',shape=[max_len])
        layer = Embedding(max_words,50,input_length=max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256,name='FC1')(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(1,name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs,outputs=layer)
        model.summary()
        model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

        model.fit(sequences_matrix,self.traindf['label'],batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
        test_sequences = tok.texts_to_sequences(self.valdf['review'])
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
        preds = model.predict(test_sequences_matrix ).round()
        self.valdf['preds'] = preds
        # self.vald_pred = [1]*self.valdf.shape[0]
        # self.valdf['preds'] = self.vald_pred
        self.base_acc = accuracy_score(self.valdf['label'],self.valdf['preds'] )
        self.base_rocauc = roc_auc_score(self.valdf['label'],self.valdf['preds'] )
        
        self.next(self.end)

        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):
        
        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))
        current.card.append(Markdown("## Examples of False Positives"))
        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0. 
        self.fp_df = self.valdf.loc[(self.valdf['preds'] == 1) & (self.valdf['label'] == 0)]
        # TODO: display the false_positives dataframe using metaflow.cards
        # current.card.append(Artifact(self.df_fp.shape[0]))
        current.card.append(
            Table.from_dataframe(self.fp_df)
        )
        # Documentation: https://docs.metaflow.org/api/cards#table
        
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false negatives predictions where the baseline is 0 and the valdf label is 1. 
        self.fn_df = self.valdf.loc[(self.valdf['preds']  == 0) & (self.valdf['label'] == 1)]
        # TODO: display the false_negatives dataframe using metaflow.cards
        # current.card.append(Artifact(self.df_fn.shape[0]))
        current.card.append(
            Table.from_dataframe(self.fn_df)
        )
if __name__ == '__main__':
    GoodFirstModelNLPFlow()


