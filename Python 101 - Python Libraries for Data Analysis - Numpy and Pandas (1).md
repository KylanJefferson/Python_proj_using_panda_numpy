# TASK #1: DEFINE SINGLE AND MULTI-DIMENSIONAL  NUMPY ARRAYS


```python
# NumPy is a Linear Algebra Library used for multidimensional arrays
# NumPy brings the best of two worlds: (1) C/Fortran computational efficiency, (2) Python language easy syntax 
import numpy as np
# Let's define a one-dimensional array 
list_1 = [50,60, 80, 100,200, 300, 500, 600]
list_1
```




    [50, 60, 80, 100, 200, 300, 500, 600]




```python
# Let's create a numpy array from the list "my_list"
my_numpy_array = np.array(list_1)
my_numpy_array
```




    array([ 50,  60,  80, 100, 200, 300, 500, 600])




```python
type(my_numpy_array)
```




    numpy.ndarray




my_matrix = np.array([  [2,5,8], [7,3,6]  ])
my_matrix

MINI CHALLENGE #1: 
- Write a code that creates the following 2x4 numpy array

```
[[3 7 9 3] 
[4 3 2 2]]
```


```python

```

# TASK #2: LEVERAGE NUMPY BUILT-IN METHODS AND FUNCTIONS 


```python
# "rand()" uniform distribution between 0 and 1
x = np.random.rand(20)
x
```




    array([2.48753892e-01, 8.62949295e-01, 3.10281406e-01, 7.38811806e-01,
           3.52716832e-04, 8.65811091e-02, 3.80123115e-01, 3.89588894e-01,
           2.84102009e-01, 7.01885914e-01, 5.25841053e-01, 9.23052791e-01,
           2.02855070e-01, 7.24640395e-01, 6.62336546e-01, 9.71629594e-01,
           8.18204448e-01, 9.97578761e-01, 8.70981773e-01, 3.28486736e-01])




```python
# you can create a matrix of random number as well
x = np.random.rand(3,3)
x
```




    array([[0.97495032, 0.45287432, 0.88804178],
           [0.22335294, 0.50687515, 0.36290399],
           [0.55703084, 0.09364501, 0.37281327]])




```python
# "randint" is used to generate random integers between upper and lower bounds
x = np.random.randint(1,50)
x
```




    23




```python
# "randint" can be used to generate a certain number of random itegers as follows
x = np.random.randint(1, 100, 15)
x
```




    array([49, 52, 14, 70, 69, 86, 70, 55, 93, 77, 10, 75, 38, 62,  5])




```python
# np.arange creates an evenly spaced values within a given interval
x = np.arange(1,30)
x
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])




```python
# create a diagonal of ones and zeros everywhere else
x = np.eye(7)
x
```




    array([[1., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 1.]])




```python
# Matrix of ones
x = np.ones((7,7))
x
```




    array([[1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1.]])




```python
# Array of zeros
x = np.zeros(8)
x
```




    array([0., 0., 0., 0., 0., 0., 0., 0.])



MINI CHALLENGE #2:
- Write a code that takes in a positive integer "x" from the user and creates a 1x10 array with random numbers ranging from 0 to "x"


```python
x =int(input('Enter a positive interger:'))

x = np.random.randint(1, x, 10)
x
```

    Enter a positive interger:2
    




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])



# TASK #3: PERFORM MATHEMATICAL OPERATIONS IN NUMPY


```python
# np.arange() returns an evenly spaced values within a given interval
x = np.arange(1,10)
x
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
y = np.arange(1, 10)
y
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# Add 2 numpy arrays together
sum = x + y
sum
```




    array([ 2,  4,  6,  8, 10, 12, 14, 16, 18])




```python
squared = x ** 2
squared
```




    array([ 1,  4,  9, 16, 25, 36, 49, 64, 81], dtype=int32)




```python
sqrt = np.sqrt(squared)
sqrt
```




    array([1., 2., 3., 4., 5., 6., 7., 8., 9.])




```python
z = np.exp(y)
z
```




    array([2.71828183e+00, 7.38905610e+00, 2.00855369e+01, 5.45981500e+01,
           1.48413159e+02, 4.03428793e+02, 1.09663316e+03, 2.98095799e+03,
           8.10308393e+03])



MINI CHALLENGE #3:
- Given the X and Y values below, obtain the distance between them

```
X = [5, 7, 20]
Y = [9, 15, 4]
```


```python
x = np.array([5, 7, 20])
y = np.array([9, 15,4])

z = np.sqrt( x**2 + y**2 )
z
```




    array([10.29563014, 16.55294536, 20.39607805])



# TASK #4: PERFORM ARRAYS SLICING AND INDEXING 


```python
my_numpy_array = np.array([3, 5, 6, 2, 8, 10, 20, 50])
my_numpy_array
```




    array([ 3,  5,  6,  2,  8, 10, 20, 50])




```python
# Access specific index from the numpy array
my_numpy_array[-1]
```




    50




```python
# Starting from the first index 0 up until and NOT including the last element
my_numpy_array[0:3]
```




    array([3, 5, 6])




```python
# Broadcasting, altering several values in a numpy array at once
my_numpy_array[0:4] = 7
my_numpy_array
```




    array([ 7,  7,  7,  7,  8, 10, 20, 50])




```python
# Let's define a two dimensional numpy array
matrix = np.random.randint(1, 10,(4, 4))
matrix
```




    array([[5, 6, 9, 4],
           [7, 4, 8, 4],
           [2, 9, 8, 7],
           [2, 4, 6, 5]])




```python
# Get a row from a mtrix
matrix[-1]
```




    array([2, 4, 6, 5])




```python
# Get one element
matrix[0][2]
```




    9



MINI CHALLENGE #4:
- In the following matrix, replace the last row with 0

```
X = [2 30 20 -2 -4]
    [3 4  40 -3 -2]
    [-3 4 -6 90 10]
    [25 45 34 22 12]
    [13 24 22 32 37]
```
X = [2 30 20 -2 -4]
    [3 4  40 -3 -2]
    [-3 4 -6 90 10]
    [25 45 34 22 12]
    [13 24 22 32 37]



```python

```

# TASK #5: PERFORM ELEMENTS SELECTION (CONDITIONAL)


```python
matrix = np.random.randint(1, 10,(5, 5))
matrix
```




    array([[7, 2, 9, 5, 6],
           [5, 8, 4, 8, 9],
           [1, 5, 7, 6, 5],
           [7, 3, 1, 5, 4],
           [2, 3, 5, 5, 6]])




```python
new_matrix = matrix[ matrix > 7  ]
new_matrix
```




    array([9, 8, 8, 9])




```python
# Obtain odd elements only
new_matrix = matrix[ matrix % 2 == 1 ]
new_matrix
```




    array([7, 9, 5, 5, 9, 1, 5, 7, 5, 7, 3, 1, 5, 3, 5, 5])



MINI CHALLENGE #5:
- In the following matrix, replace negative elements by 0 and replace odd elements with -2


```
X = [2 30 20 -2 -4]
    [3 4  40 -3 -2]
    [-3 4 -6 90 10]
    [25 45 34 22 12]
    [13 24 22 32 37]
```



```python

```

# TASK #6: UNDERSTAND PANDAS FUNDAMENTALS


```python
# Pandas is a data manipulation and analysis tool that is built on Numpy.
# Pandas uses a data structure known as DataFrame (think of it as Microsoft excel in Python). 
# DataFrames empower programmers to store and manipulate data in a tabular fashion (rows and columns).
# Series Vs. DataFrame? Series is considered a single column of a DataFrame.
```


```python
import pandas as pd
```


```python
# Let's define a two-dimensional Pandas DataFrame
# Note that you can create a pandas dataframe from a python dictionary
bank_client_df = pd.DataFrame({'Bank Client ID':[111, 222, 333, 444],
                              'Bank Client Name':['Chanel', 'Steve',  'Mitch', 'Ryan'],
                              'Net Worth [$]':[3500, 29000, 10000, 2000],
                              'Years with Bank':[3, 4, 9, 5]})
bank_client_df
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's obtain the data type 
type(bank_client_df)

```




    pandas.core.frame.DataFrame




```python
# you can only view the first couple of rows using .head()
bank_client_df.head(2)
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# you can only view the last couple of rows using .tail()
bank_client_df.tail(2)
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



MINI CHALLENGE #6:
- A porfolio contains a collection of securities such as stocks, bonds and ETFs. Define a dataframe named 'portfolio_df' that holds 3 different stock ticker symbols, number of shares, and price per share (feel free to choose any stocks)
- Calculate the total value of the porfolio including all stocks


```python

```

# TASK #7: PANDAS WITH CSV AND HTML DATA


```python
# Pandas is used to read a csv file and store data in a DataFrame

```


```python

```


```python
# Read tabular data using read_html
house_price_df = pd.read_html('https://www.livingin-canada.com/house-prices-canada.html')
house_price_df[0]

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
      <th>City</th>
      <th>Average House Price</th>
      <th>12 Month Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Vancouver, BC</td>
      <td>$1,036,000</td>
      <td>+ 2.63 %</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toronto, Ont</td>
      <td>$870,000</td>
      <td>+10.2 %</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ottawa, Ont</td>
      <td>$479,000</td>
      <td>+ 15.4 %</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Calgary, Alb</td>
      <td>$410,000</td>
      <td>– 1.5 %</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Montreal, Que</td>
      <td>$435,000</td>
      <td>+ 9.3 %</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Halifax, NS</td>
      <td>$331,000</td>
      <td>+ 3.6 %</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Regina, Sask</td>
      <td>$254,000</td>
      <td>– 3.9 %</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Fredericton, NB</td>
      <td>$198,000</td>
      <td>– 4.3 %</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(adsbygoogle = window.adsbygoogle || []).push(...</td>
      <td>(adsbygoogle = window.adsbygoogle || []).push(...</td>
      <td>(adsbygoogle = window.adsbygoogle || []).push(...</td>
    </tr>
  </tbody>
</table>
</div>




```python
pip install lxml
```

    Requirement already satisfied: lxml in c:\users\administrator\appdata\local\programs\python\python37\lib\site-packages (4.9.1)
    Note: you may need to restart the kernel to use updated packages.
    

    tensorboard 2.1.1 has requirement setuptools>=41.0.0, but you'll have setuptools 39.0.1 which is incompatible.
    google-auth 1.12.0 has requirement setuptools>=40.3.0, but you'll have setuptools 39.0.1 which is incompatible.
    You are using pip version 10.0.1, however version 22.2.2 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    


```python
python -m pip install --upgrade pip
```


      File "<ipython-input-39-0a8cf7b99133>", line 1
        python -m pip install --upgrade pip
                    ^
    SyntaxError: invalid syntax
    


MINI CHALLENGE #7:
- Write a code that uses Pandas to read tabular US retirement data
- You can use data from here: https://www.ssa.gov/oact/progdata/nra.html 


```python
gov_page= pd.read_html('https://www.ssa.gov/oact/progdata/nra.html')
gov_page[0]
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
      <th>Year of birth</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1937 and prior</td>
      <td>65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1938</td>
      <td>65 and 2 months</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1939</td>
      <td>65 and 4 months</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1940</td>
      <td>65 and 6 months</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1941</td>
      <td>65 and 8 months</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1942</td>
      <td>65 and 10 months</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1943-54</td>
      <td>66</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1955</td>
      <td>66 and 2 months</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1956</td>
      <td>66 and 4 months</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1957</td>
      <td>66 and 6 months</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1958</td>
      <td>66 and 8 months</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1959</td>
      <td>66 and 10 months</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1960 and later</td>
      <td>67</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Notes: 1. Persons born on January 1 of any yea...</td>
      <td>Notes: 1. Persons born on January 1 of any yea...</td>
    </tr>
  </tbody>
</table>
</div>



# TASK #8: PANDAS OPERATIONS


```python
# Let's define a dataframe as follows:
bank_client_df = pd.DataFrame({'Bank Client ID':[111, 222, 333, 444],
                              'Bank Client Name':['Chanel', 'Steve',  'Mitch', 'Ryan'],
                              'Net Worth [$]':[3500, 29000, 10000, 2000],
                              'Years with Bank':[3, 4, 9, 5]})
bank_client_df
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pick certain rows that satisfy a certain criteria 
df_loyal = bank_client_df[  bank_client_df['Years with Bank'] >= 5    ]
df_loyal
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Delete a column from a DataFrame
del bank_client_df['Bank Client ID']
bank_client_df
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
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



MINI CHALLENGE #8:
- Using "bank_client_df" DataFrame, leverage pandas operations to only select high networth individuals with minimum $5000 
- What is the combined networth for all customers with 5000+ networth?


```python

```

# TASK #9: PANDAS WITH FUNCTIONS


```python
# Let's define a dataframe as follows:
bank_client_df = pd.DataFrame({'Bank Client ID':[111, 222, 333, 444],
                              'Bank Client Name':['Chanel', 'Steve',  'Mitch', 'Ryan'],
                              'Net Worth [$]':[3500, 29000, 10000, 2000],
                              'Years with Bank':[3, 4, 9, 5]})
bank_client_df
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
      <th>Bank Client ID</th>
      <th>Bank Client Name</th>
      <th>Net Worth [$]</th>
      <th>Years with Bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define a function that increases all clients networth (stocks) by a fixed value of 20% (for simplicity sake) 
def networth_update(balance):
    return balance *3 +200
```


```python
# You can apply a function to the DataFrame 
results = bank_client_df['Net Worth [$]'].apply(networth_update)

```


```python
results
```




    0    10700
    1    87200
    2    30200
    3     6200
    Name: Net Worth [$], dtype: int64




```python
results.sum()
```




    134300



MINI CHALLENGE #9:
- Define a function that triples the stock prices and adds $200
- Apply the function to the DataFrame
- Calculate the updated total networth of all clients combined


```python

```

# TASK #10: PERFORM SORTING AND ORDERING IN PANDAS


```python
# Let's define a dataframe as follows:
bank_client_df = pd.DataFrame({'Bank client ID':[111, 222, 333, 444], 
                               'Bank Client Name':['Chanel', 'Steve', 'Mitch', 'Ryan'], 
                               'Net worth [$]':[3500, 29000, 10000, 2000], 
                               'Years with bank':[3, 4, 9, 5]})
bank_client_df
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
      <th>Bank client ID</th>
      <th>Bank Client Name</th>
      <th>Net worth [$]</th>
      <th>Years with bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# You can sort the values in the dataframe according to number of years with bank
bank_client_df.sort_values(by = 'Years with bank')
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
      <th>Bank client ID</th>
      <th>Bank Client Name</th>
      <th>Net worth [$]</th>
      <th>Years with bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Note that nothing changed in memory! you have to make sure that inplace is set to True

```


```python
# Set inplace = True to ensure that change has taken place in memory 
bank_client_df.sort_values(by = 'Years with bank', inplace = True)
```


```python
# Note that now the change (ordering) took place 
bank_client_df
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
      <th>Bank client ID</th>
      <th>Bank Client Name</th>
      <th>Net worth [$]</th>
      <th>Years with bank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>111</td>
      <td>Chanel</td>
      <td>3500</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>222</td>
      <td>Steve</td>
      <td>29000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>444</td>
      <td>Ryan</td>
      <td>2000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333</td>
      <td>Mitch</td>
      <td>10000</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



# TASK #11: PERFORM CONCATENATING AND MERGING WITH PANDAS


```python
# Check this out: https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
```


```python
df1 = pd.DataFrame({ 'A': ['A0', 'A1', 'A2', 'A3'],
                   'B':['B0', 'B1', 'B2', 'B3'],
                   'C':['C0', 'C1', 'C2', 'C3'],
                   'D':['D0', 'D1', 'D2', 'D3']}, 
                  index = [0,1,2,3])
```


```python
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame({ 'A': ['A4', 'A5', 'A6', 'A7'],
                   'B':['B4', 'B5', 'B6', 'B7'],
                   'C':['C4', 'C5', 'C6', 'C7'],
                   'D':['D4', 'D5', 'D6', 'D7']}, 
                  index = [0,1,2,3])
```


```python
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df3 = pd.DataFrame({'A':['A8', 'A9', 'A10', 'A11'],
                    'B':['B8', 'B9', 'B10', 'B11'],
                    'C':['C8', 'C9', 'C10', 'C11'],
                    'D':['D8', 'D9', 'D10', 'D11']}, 
                  index = [8,9,10,11])
```


```python
df3
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([df1, df2, df3])
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A0</td>
      <td>B0</td>
      <td>C0</td>
      <td>D0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A1</td>
      <td>B1</td>
      <td>C1</td>
      <td>D1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A2</td>
      <td>B2</td>
      <td>C2</td>
      <td>D2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A3</td>
      <td>B3</td>
      <td>C3</td>
      <td>D3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>A4</td>
      <td>B4</td>
      <td>C4</td>
      <td>D4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A5</td>
      <td>B5</td>
      <td>C5</td>
      <td>D5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A6</td>
      <td>B6</td>
      <td>C6</td>
      <td>D6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A7</td>
      <td>B7</td>
      <td>C7</td>
      <td>D7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A8</td>
      <td>B8</td>
      <td>C8</td>
      <td>D8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A9</td>
      <td>B9</td>
      <td>C9</td>
      <td>D9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A10</td>
      <td>B10</td>
      <td>C10</td>
      <td>D10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A11</td>
      <td>B11</td>
      <td>C11</td>
      <td>D11</td>
    </tr>
  </tbody>
</table>
</div>



# TASK #12: PROJECT AND CONCLUDING REMARKS

- Define a dataframe named 'Bank_df_1' that contains the first and last names for 5 bank clients with IDs = 1, 2, 3, 4, 5 
- Assume that the bank got 5 new clients, define another dataframe named 'Bank_df_2' that contains a new clients with IDs = 6, 7, 8, 9, 10
- Let's assume we obtained additional information (Annual Salary) about all our bank customers (10 customers) 
- Concatenate both 'bank_df_1' and 'bank_df_2' dataframes
- Merge client names and their newly added salary information using the 'Bank Client ID'
- Let's assume that you became a new client to the bank
- Define a new DataFrame that contains your information such as client ID (choose 11), first name, last name, and annual salary.
- Add this new dataframe to the original dataframe 'bank_df_all'.


```python
raw_data = { 'Bank Client ID': ['1', '2', '3', '4', '5'],
           'First Name':['Nancy', 'Alex', 'Shep', 'Max', 'Allen'],
           'Last name':['Rob', 'Ali', 'George', 'Mitch', 'Steve']}
Bank_df_1 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_1
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Nancy</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Alex</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Shep</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Max</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Allen</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_data = { 'Bank Client ID': ['6', '7', '8', '9', '10'],
           'First Name':['Bill', 'Dina', 'Sarah', 'Heather', 'Holly'],
           'Last name':['Christian', 'Steve', 'Rob', 'Mo', 'Michelle']}
Bank_df_2 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_2
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
raw_data = {'Bank Client ID': ['1','2','3','4','5','6','7','8','9','10'],
           'Annual Salary [$/year]': [25000, 35000, 45000, 48000, 49000, 32000, 33000, 34000, 23000, 22000]}
bank_df_salary = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'Annual Salary [$/year]'])
bank_df_salary
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
      <th>Bank Client ID</th>
      <th>Annual Salary [$/year]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>25000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>48000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>49000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>34000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>22000</td>
    </tr>
  </tbody>
</table>
</div>




```python
bank_df_all = pd.concat([Bank_df_1 , Bank_df_2])
bank_df_all
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
bank_df_all = pd.merge(bank_df_all, bank_df_salary, on = 'Bank Client ID')
bank_df_all
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
      <th>Annual Salary [$/year]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
      <td>32000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
      <td>34000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
      <td>34000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
      <td>22000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
      <td>22000</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_client = {'Bank Client ID': ['11'],
              'First Name': ['Ryan'],
              'Last Name': ['Ahmed'],
              'Annual Salary [$/year]': [5000]}
new_client_df = pd.DataFrame(new_client, columns = ['Bank Client ID', 'First Name', 'Last Name', 'Annual Salary[$/year]'])

```


```python
new_client_df
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
      <th>Annual Salary[$/year]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>Ryan</td>
      <td>Ahmed</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_df = pd.concat([bank_df_all, new_client_df], axis = 0)
new_df
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
      <th>Bank Client ID</th>
      <th>First Name</th>
      <th>Last Name</th>
      <th>Annual Salary [$/year]</th>
      <th>Annual Salary[$/year]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
      <td>32000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Bill</td>
      <td>NaN</td>
      <td>32000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
      <td>33000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>Dina</td>
      <td>NaN</td>
      <td>33000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
      <td>34000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>Sarah</td>
      <td>NaN</td>
      <td>34000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
      <td>23000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>Heather</td>
      <td>NaN</td>
      <td>23000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
      <td>22000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Holly</td>
      <td>NaN</td>
      <td>22000.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>Ryan</td>
      <td>Ahmed</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# EXCELLENT JOB!

# MINI CHALLENGES SOLUTIONS

MINI CHALLENGE #1 SOLUTION: 
- Write a code that creates the following 2x4 numpy array

```
[[3 7 9 3] 
[4 3 2 2]]
```


```python
x = np.array([[[3, 7, 9, 3] , [4, 3, 2, 2]]])
x
```

MINI CHALLENGE #2 SOLUTION:
- Write a code that takes in a positive integer "x" from the user and creates a 1x10 array with random numbers ranging from 0 to "x"


```python
x = int(input("Please enter a positive integer value: "))
x = np.random.randint(1, x, 10)
x
```


```python

```

MINI CHALLENGE #3 SOLUTION:
- Given the X and Y values below, obtain the distance between them


```
X = [5, 7, 20]
Y = [9, 15, 4]
```





```python
X = np.array([5, 7, 20])
Y = np.array([9, 15, 4])
Z = np.sqrt(X**2 + Y**2)
Z
```

MINI CHALLENGE #4 SOLUTION:
- In the following matrix, replace the last row with 0

```
X = [2 30 20 -2 -4]
    [3 4  40 -3 -2]
    [-3 4 -6 90 10]
    [25 45 34 22 12]
    [13 24 22 32 37]
```





```python
X = np.array([[2, 30, 20, -2, -4],
    [3, 4,  40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]])

```


```python
X[4] = 0
X
```

MINI CHALLENGE #5 SOLUTION:
- In the following matrix, replace negative elements by 0 and replace odd elements with -2


```
X = [2 30 20 -2 -4]
    [3 4  40 -3 -2]
    [-3 4 -6 90 10]
    [25 45 34 22 12]
    [13 24 22 32 37]
```


```python
X = np.array([[2, 30, 20, -2, -4],
    [3, 4,  40, -3, -2],
    [-3, 4, -6, 90, 10],
    [25, 45, 34, 22, 12],
    [13, 24, 22, 32, 37]])

X[X<0] = 0
X[X%2==1] = -2
X
```

MINI CHALLENGE #6 SOLUTION:
- A porfolio contains a collection of securities such as stocks, bonds and ETFs. Define a dataframe named 'portfolio_df' that holds 3 different stock ticker symbols, number of shares, and price per share (feel free to choose any stocks)
- Calculate the total value of the porfolio including all stocks


```python
portfolio_df = pd.DataFrame({'stock ticker symbols':['AAPL', 'AMZN', 'T'],
                             'price per share [$]':[3500, 200, 40], 
                             'Number of stocks':[3, 4, 9]})
portfolio_df
```


```python
stocks_dollar_value = portfolio_df['price per share [$]'] * portfolio_df['Number of stocks']
print(stocks_dollar_value)
print('Total portfolio value = {}'.format(stocks_dollar_value.sum()))
```

MINI CHALLENGE #7 SOLUTION:
- Write a code that uses Pandas to read tabular US retirement data
- You can use data from here: https://www.ssa.gov/oact/progdata/nra.html 


```python
# Read tabular data using read_html
retirement_age_df = pd.read_html('https://www.ssa.gov/oact/progdata/nra.html')
retirement_age_df
```

MINI CHALLENGE #8 SOLUTION:
- Using "bank_client_df" DataFrame, leverage pandas operations to only select high networth individuals with minimum $5000 
- What is the combined networth for all customers with 5000+ networth?


```python
df_high_networth = bank_client_df[ (bank_client_df['Net worth [$]'] >= 5000) ]
df_high_networth
```


```python
df_high_networth['Net worth [$]'].sum()
```

MINI CHALLENGE #9 SOLUTION:
- Define a function that triples the stock prices and adds $200
- Apply the function to the DataFrame
- Calculate the updated total networth of all clients combined


```python
def networth_update(balance):
    return balance * 3 + 200 
```


```python
# You can apply a function to the DataFrame 
results = bank_client_df['Net worth [$]'].apply(networth_update)
results
```


```python
results.sum()
```

PROJECT SOLUTION:


```python
# Creating a dataframe from a dictionary
# Let's define a dataframe with a list of bank clients with IDs = 1, 2, 3, 4, 5 

raw_data = {'Bank Client ID': ['1', '2', '3', '4', '5'],
            'First Name': ['Nancy', 'Alex', 'Shep', 'Max', 'Allen'], 
            'Last Name': ['Rob', 'Ali', 'George', 'Mitch', 'Steve']}

Bank_df_1 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_1


# Let's define another dataframe for a separate list of clients (IDs = 6, 7, 8, 9, 10)
raw_data = {
        'Bank Client ID': ['6', '7', '8', '9', '10'],
        'First Name': ['Bill', 'Dina', 'Sarah', 'Heather', 'Holly'], 
        'Last Name': ['Christian', 'Mo', 'Steve', 'Bob', 'Michelle']}
Bank_df_2 = pd.DataFrame(raw_data, columns = ['Bank Client ID', 'First Name', 'Last Name'])
Bank_df_2


# Let's assume we obtained additional information (Annual Salary) about our bank customers 
# Note that data obtained is for all clients with IDs 1 to 10 
raw_data = {
        'Bank Client ID': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'Annual Salary [$/year]': [25000, 35000, 45000, 48000, 49000, 32000, 33000, 34000, 23000, 22000]}
bank_df_salary = pd.DataFrame(raw_data, columns = ['Bank Client ID','Annual Salary [$/year]'])
bank_df_salary


# Let's concatenate both dataframes #1 and #2
# Note that we now have client IDs from 1 to 10
bank_df_all = pd.concat([Bank_df_1, Bank_df_2])
bank_df_all


# Let's merge all data on 'Bank Client ID'
bank_df_all = pd.merge(bank_df_all, bank_df_salary, on = 'Bank Client ID')
bank_df_all

```


```python
new_client = {
        'Bank Client ID': ['11'],
        'First Name': ['Ry'], 
        'Last Name': ['Aly'],
        'Annual Salary [$/year]' : [1000]}
new_client_df = pd.DataFrame(new_client, columns = ['Bank Client ID', 'First Name', 'Last Name', 'Annual Salary [$/year]'])
new_client_df
```


```python
new_df = pd.concat([bank_df_all, new_client_df], axis = 0)
new_df
```
