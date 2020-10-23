# Credit-card-linear-regression

<h1>Prerequisites:</h1><br>
You will need the following programmes properly installed on your computer.<br>
Python 3.7+ <br>
django 2.2+ <br>

<h3>Virtual Environment</h3><br>
To install virtual environment on your system use:<br>
for windows:

```shell
pip install virtualenv
```
<br>
for linux(for python 3 and above)

```shell
pip3 install virtualenv
```

<h1>Installation and Running :</h1><br>

```shell
git clone https://github.com/ongraphpythondev/videoprocessing_moviepy_POC/
cd videoprocessing_moviepy_POC
```

for windows:
```shell
virtualenv venv
venv\Scripts\activate
```
for linux(for python 3 and above):
```shell
virtualenv venv -p python3
source venv/bin/activate
```
<br>
<h1>Install required packages for the project to run</h1>

```shell
pip install -r requirements.txt
python stat_reg.py
```

<br>
<h1>Objective :</h1><br>

To understand what's driving the total spend (Primary Credit Card +
Secondary Credit Card) of the customer. Given the factors, predict credit limit for the new applicants.

<h3>Libraries Used:</h3><br>

- pandas
- numpy
- scipy
- sklearn
- statsmodels
<br>

<h3>Conclusion :</h3><br>
Linear regression is used to train the model. The following features were selected to train the model i.e. these features are driving the total spend of the customer:

- age : Age in years 
- carcatvalue : Primary vehicle price category
- card2 : Secondary credit card
- card2type : Designation of secondary credit card
- card : Primary credit card
- cardtype : Designation of primary credit card
- debtinc : Debt to income ratio (x100)
- lncreddebt : Log-credit card debt
- lninc : Log-income
- othdebt : Other debt in thousands

