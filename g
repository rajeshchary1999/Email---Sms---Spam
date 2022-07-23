if st.button('predict'):
   #1 preprocess
    transformed_sms = transform_text(input_sms)

   #2vectorize
   vector_input = tfidf.transform([transformed_sms])

   ## predict
   result  = model.predict(vector_input)[0]

   #4 Disply
     if result == 1:

         st.header("Spam")
     else:
          st.header("Not Spam")