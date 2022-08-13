#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
# flask 5000
# jupyter 8888


# In[2]:


import joblib


# In[3]:


app = Flask(__name__) #function loaded into object
# __name__ is a way for python to confirm author


# In[4]:


@app.route("/", methods=["GET", "POST"]) # python declarator 
def index():
    if request.method == "POST":
        rates = float(request.form.get("rates"))
        print(rates)
        model1 = joblib.load("regression_DBS")
        pred1 = model1.predict([[rates]])
        model2 = joblib.load("tree_DBS")
        pred2 = model2.predict([[rates]])
        return(render_template("index.html", result1=pred1, result2=pred2))
    else:
        return(render_template("index.html", result1="waiting", result2="waiting"))


# In[5]:


if __name__ == "__main__": 
    app.run()


# In[ ]:




