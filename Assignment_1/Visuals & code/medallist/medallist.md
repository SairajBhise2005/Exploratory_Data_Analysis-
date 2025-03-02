```python
import pandas as pd

# Load the CSV file into a DataFrame
file_path ="C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"  # Path to your uploaded file
df = pd.read_csv(file_path)

# Display the first 5 rows of the DataFrame
df.head()

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
      <th>medal_date</th>
      <th>medal_type</th>
      <th>medal_code</th>
      <th>name</th>
      <th>gender</th>
      <th>country_code</th>
      <th>country</th>
      <th>country_long</th>
      <th>nationality</th>
      <th>team</th>
      <th>team_gender</th>
      <th>discipline</th>
      <th>event</th>
      <th>event_type</th>
      <th>url_event</th>
      <th>birth_date</th>
      <th>code_athlete</th>
      <th>code_team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-07-27</td>
      <td>Gold Medal</td>
      <td>1.0</td>
      <td>EVENEPOEL Remco</td>
      <td>Male</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>Belgium</td>
      <td>Belgium</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cycling Road</td>
      <td>Men's Individual Time Trial</td>
      <td>ATH</td>
      <td>/en/paris-2024/results/cycling-road/men-s-indi...</td>
      <td>2000-01-25</td>
      <td>1903136</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-07-27</td>
      <td>Silver Medal</td>
      <td>2.0</td>
      <td>GANNA Filippo</td>
      <td>Male</td>
      <td>ITA</td>
      <td>Italy</td>
      <td>Italy</td>
      <td>Italy</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cycling Road</td>
      <td>Men's Individual Time Trial</td>
      <td>ATH</td>
      <td>/en/paris-2024/results/cycling-road/men-s-indi...</td>
      <td>1996-07-25</td>
      <td>1923520</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-07-27</td>
      <td>Bronze Medal</td>
      <td>3.0</td>
      <td>van AERT Wout</td>
      <td>Male</td>
      <td>BEL</td>
      <td>Belgium</td>
      <td>Belgium</td>
      <td>Belgium</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cycling Road</td>
      <td>Men's Individual Time Trial</td>
      <td>ATH</td>
      <td>/en/paris-2024/results/cycling-road/men-s-indi...</td>
      <td>1994-09-15</td>
      <td>1903147</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-07-27</td>
      <td>Gold Medal</td>
      <td>1.0</td>
      <td>BROWN Grace</td>
      <td>Female</td>
      <td>AUS</td>
      <td>Australia</td>
      <td>Australia</td>
      <td>Australia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cycling Road</td>
      <td>Women's Individual Time Trial</td>
      <td>ATH</td>
      <td>/en/paris-2024/results/cycling-road/women-s-in...</td>
      <td>1992-07-07</td>
      <td>1940173</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-07-27</td>
      <td>Silver Medal</td>
      <td>2.0</td>
      <td>HENDERSON Anna</td>
      <td>Female</td>
      <td>GBR</td>
      <td>Great Britain</td>
      <td>Great Britain</td>
      <td>Great Britain</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cycling Road</td>
      <td>Women's Individual Time Trial</td>
      <td>ATH</td>
      <td>/en/paris-2024/results/cycling-road/women-s-in...</td>
      <td>1998-11-14</td>
      <td>1912525</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd                                                         #2

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop any rows with missing country or gender data
cleaned_data = medal_data.dropna(subset=['country', 'gender'])

# Group the data by country and gender to count the number of medals won by each
country_gender_representation = cleaned_data.groupby(['country', 'gender']).size().unstack().fillna(0)

# Rename the columns for better clarity
country_gender_representation.columns = ['Female Medals', 'Male Medals']

# Sort the result by the total number of medals (both male and female)
country_gender_representation['Total Medals'] = country_gender_representation['Female Medals'] + country_gender_representation['Male Medals']
sorted_representation = country_gender_representation.sort_values(by='Total Medals', ascending=False)

# Display the top countries by male and female medalists
sorted_representation.head()

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
      <th>Female Medals</th>
      <th>Male Medals</th>
      <th>Total Medals</th>
    </tr>
    <tr>
      <th>country</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>United States</th>
      <td>187.0</td>
      <td>143.0</td>
      <td>330.0</td>
    </tr>
    <tr>
      <th>France</th>
      <td>61.0</td>
      <td>125.0</td>
      <td>186.0</td>
    </tr>
    <tr>
      <th>China</th>
      <td>112.0</td>
      <td>56.0</td>
      <td>168.0</td>
    </tr>
    <tr>
      <th>Great Britain</th>
      <td>81.0</td>
      <td>80.0</td>
      <td>161.0</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>76.0</td>
      <td>47.0</td>
      <td>123.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pandas as pd                                                         #6
import numpy as np

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop rows with missing birth_date or gender information
medal_data_clean = medal_data.dropna(subset=['birth_date', 'gender'])

# Convert birth_date to datetime format
medal_data_clean['birth_date'] = pd.to_datetime(medal_data_clean['birth_date'])

# Extract year of birth and calculate age (assuming the event year is 2024)
medal_data_clean['age'] = 2024 - medal_data_clean['birth_date'].dt.year

# Group by gender to calculate average and median age
age_by_gender = medal_data_clean.groupby('gender')['age'].agg(['mean', 'median', 'std'])

# Display the age statistics for male and female medalists
print(age_by_gender)

# Optional: Visualize the age distribution using a boxplot (if you have matplotlib installed)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(x='gender', y='age', data=medal_data_clean)
plt.title('Age Distribution of Male and Female Medalists')
plt.show()

```

    C:\Users\agaje\AppData\Local\Temp\ipykernel_10948\764579963.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      medal_data_clean['birth_date'] = pd.to_datetime(medal_data_clean['birth_date'])
    C:\Users\agaje\AppData\Local\Temp\ipykernel_10948\764579963.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      medal_data_clean['age'] = 2024 - medal_data_clean['birth_date'].dt.year
    

                 mean  median       std
    gender                             
    Female  26.482788    26.0  4.860160
    Male    27.286087    27.0  5.015457
    


    
![png](output_2_2.png)
    



```python
import pandas as pd                                                     # 1

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop rows with missing gender or medal_type information
cleaned_data = medal_data.dropna(subset=['gender', 'medal_type'])

# Group the data by gender and medal type to count the number of each medal won by male and female athletes
medal_distribution = cleaned_data.groupby(['gender', 'medal_type']).size().unstack().fillna(0)

# Rename the columns for better clarity
medal_distribution.columns = ['Bronze Medals', 'Gold Medals', 'Silver Medals']

# Display the medal distribution
print(medal_distribution)

# Optional: Visualize the medal distribution using a bar plot (if you have matplotlib installed)
import matplotlib.pyplot as plt

medal_distribution.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Medal Distribution between Male and Female Athletes')
plt.ylabel('Number of Medals')
plt.xticks(rotation=0)
plt.show()

```

            Bronze Medals  Gold Medals  Silver Medals
    gender                                           
    Female            402          378            382
    Male              403          373            374
    


    
![png](output_3_1.png)
    



```python
import pandas as pd                                     #4

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop rows with missing gender or event information
cleaned_data = medal_data.dropna(subset=['gender', 'event'])

# Group the data by event and gender, then count unique events for each gender
event_count_by_gender = cleaned_data.groupby('gender')['event'].nunique()

# Display the number of unique events for male and female athletes
print(event_count_by_gender)

# Calculate the difference between male and female events
difference = event_count_by_gender['Male'] - event_count_by_gender['Female']
print(f"There are {difference} more events where male athletes won medals compared to female athletes." if difference > 0 else f"There are {abs(difference)} more events where female athletes won medals compared to male athletes.")

```

    gender
    Female    150
    Male      155
    Name: event, dtype: int64
    There are 5 more events where male athletes won medals compared to female athletes.
    


```python
import pandas as pd                              #3

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop rows with missing gender or discipline information
cleaned_data = medal_data.dropna(subset=['gender', 'discipline'])

# Group the data by discipline and gender to count the number of medals won in each discipline by gender
discipline_gender_distribution = cleaned_data.groupby(['discipline', 'gender']).size().unstack().fillna(0)

# Rename the columns for better clarity
discipline_gender_distribution.columns = ['Female Medals', 'Male Medals']

# Calculate the total number of medals in each discipline
discipline_gender_distribution['Total Medals'] = discipline_gender_distribution['Female Medals'] + discipline_gender_distribution['Male Medals']

# Calculate the percentage of medals won by each gender
discipline_gender_distribution['Female Percentage'] = (discipline_gender_distribution['Female Medals'] / discipline_gender_distribution['Total Medals']) * 100
discipline_gender_distribution['Male Percentage'] = (discipline_gender_distribution['Male Medals'] / discipline_gender_distribution['Total Medals']) * 100

# Sort the disciplines by the difference in percentage between male and female medals
discipline_gender_distribution['Dominance Difference'] = abs(discipline_gender_distribution['Female Percentage'] - discipline_gender_distribution['Male Percentage'])

# Sort disciplines by dominance difference and display the top results
dominance_by_gender = discipline_gender_distribution.sort_values(by='Dominance Difference', ascending=False)

# Display the top disciplines where one gender dominates
print(dominance_by_gender[['Female Medals', 'Male Medals', 'Female Percentage', 'Male Percentage', 'Dominance Difference']].head())

# Optional: Visualize the gender dominance in sports disciplines using a bar plot (if you have matplotlib installed)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
dominance_by_gender[['Female Percentage', 'Male Percentage']].head(10).plot(kind='bar', stacked=True)
plt.title('Top 10 Disciplines with Gender Dominance')
plt.ylabel('Percentage of Medals')
plt.xticks(rotation=45)
plt.show()

```

                         Female Medals  Male Medals  Female Percentage  \
    discipline                                                           
    Artistic Swimming             33.0          0.0         100.000000   
    Rhythmic Gymnastics           18.0          0.0         100.000000   
    Wrestling                     24.0         48.0          33.333333   
    Equestrian                    15.0         29.0          34.090909   
    Artistic Gymnastics           30.0         37.0          44.776119   
    
                         Male Percentage  Dominance Difference  
    discipline                                                  
    Artistic Swimming           0.000000            100.000000  
    Rhythmic Gymnastics         0.000000            100.000000  
    Wrestling                  66.666667             33.333333  
    Equestrian                 65.909091             31.818182  
    Artistic Gymnastics        55.223881             10.447761  
    


    <Figure size 1000x600 with 0 Axes>



    
![png](output_5_2.png)
    



```python
import pandas as pd                                                          #1
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected file path with double backslashes
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a pivot table to analyze gender distribution by sport
gender_sport_distribution = df.pivot_table(index='discipline', columns='gender', aggfunc='size', fill_value=0)

# Reset the index for easier plotting
gender_sport_distribution = gender_sport_distribution.reset_index()

# Melt the DataFrame to long format for seaborn plotting
gender_sport_distribution_melted = gender_sport_distribution.melt(id_vars='discipline', var_name='gender', value_name='Count')

# Plot the data using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='discipline', y='Count', hue='gender', data=gender_sport_distribution_melted, palette='coolwarm')

# Rotate the x-axis labels for readability
plt.xticks(rotation=90)

# Add labels and title
plt.title('Gender Dominance in Different Sports Disciplines', fontsize=16)
plt.xlabel('discipline', fontsize=12)
plt.ylabel('Number of Athletes', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_6_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected file path with double backslashes
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Check column names to ensure 'Event' and 'Gender' are correct
print(df.columns)

# Group by Event and Gender, then count unique events for each gender
events_by_gender = df.groupby('gender')['event'].nunique()

# Convert to DataFrame for easier plotting
events_by_gender_df = events_by_gender.reset_index(name='Event Count')

# Plot the data using seaborn
plt.figure(figsize=(8, 6))
sns.barplot(x='gender', y='Event Count', data=events_by_gender_df, palette='Set2')

# Add labels and title
plt.title('Comparison of Events with Medal Wins by Male and Female Athletes', fontsize=16)
plt.xlabel('gender', fontsize=12)
plt.ylabel('Number of Unique Events', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
```

    Index(['medal_date', 'medal_type', 'medal_code', 'name', 'gender',
           'country_code', 'country', 'country_long', 'nationality', 'team',
           'team_gender', 'discipline', 'event', 'event_type', 'url_event',
           'birth_date', 'code_athlete', 'code_team'],
          dtype='object')
    

    C:\Users\agaje\AppData\Local\Temp\ipykernel_10948\4154264139.py:22: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x='gender', y='Event Count', data=events_by_gender_df, palette='Set2')
    


    
![png](output_7_2.png)
    



```python
import pandas as pd

# Load the CSV file (replace 'medallists.csv' with your file path)
medal_data = pd.read_csv("C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv")

# Drop rows with missing gender or discipline information
cleaned_data = medal_data.dropna(subset=['gender', 'discipline'])

# Group the data by discipline and gender to count the number of medals won in each discipline by gender
discipline_gender_distribution = cleaned_data.groupby(['discipline', 'gender']).size().unstack().fillna(0)

# Rename the columns for better clarity
discipline_gender_distribution.columns = ['Female Medals', 'Male Medals']

# Calculate the total number of medals in each discipline
discipline_gender_distribution['Total Medals'] = discipline_gender_distribution['Female Medals'] + discipline_gender_distribution['Male Medals']

# Calculate the percentage of medals won by each gender
discipline_gender_distribution['Female Percentage'] = (discipline_gender_distribution['Female Medals'] / discipline_gender_distribution['Total Medals']) * 100
discipline_gender_distribution['Male Percentage'] = (discipline_gender_distribution['Male Medals'] / discipline_gender_distribution['Total Medals']) * 100

# Sort the disciplines by the difference in percentage between male and female medals
discipline_gender_distribution['Dominance Difference'] = abs(discipline_gender_distribution['Female Percentage'] - discipline_gender_distribution['Male Percentage'])

# Sort disciplines by dominance difference and display the top results
dominance_by_gender = discipline_gender_distribution.sort_values(by='Dominance Difference', ascending=False)

# Display the top disciplines where one gender dominates
print(dominance_by_gender[['Female Medals', 'Male Medals', 'Female Percentage', 'Male Percentage', 'Dominance Difference']].head())

# Optional: Visualize the gender dominance in sports disciplines using a bar plot (if you have matplotlib installed)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
dominance_by_gender[['Female Percentage', 'Male Percentage']].head(10).plot(kind='bar', stacked=True)
plt.title('Top 10 Disciplines with Gender Dominance')
plt.ylabel('Percentage of Medals')
plt.xticks(rotation=45)
plt.show()

```

                         Female Medals  Male Medals  Female Percentage  \
    discipline                                                           
    Artistic Swimming             33.0          0.0         100.000000   
    Rhythmic Gymnastics           18.0          0.0         100.000000   
    Wrestling                     24.0         48.0          33.333333   
    Equestrian                    15.0         29.0          34.090909   
    Artistic Gymnastics           30.0         37.0          44.776119   
    
                         Male Percentage  Dominance Difference  
    discipline                                                  
    Artistic Swimming           0.000000            100.000000  
    Rhythmic Gymnastics         0.000000            100.000000  
    Wrestling                  66.666667             33.333333  
    Equestrian                 65.909091             31.818182  
    Artistic Gymnastics        55.223881             10.447761  
    


    <Figure size 1000x600 with 0 Axes>



    
![png](output_8_2.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected file path with double backslashes
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a pivot table to analyze gender distribution by sport
gender_sport_distribution = df.pivot_table(index='discipline', columns='gender', aggfunc='size', fill_value=0)

# Reset the index for easier plotting
gender_sport_distribution = gender_sport_distribution.reset_index()

# Calculate total counts for sorting purposes
gender_sport_distribution['Total'] = gender_sport_distribution.sum(axis=1)

# Sort by total count in ascending order
gender_sport_distribution_sorted = gender_sport_distribution.sort_values(by='Total', ascending=True)

# Melt the DataFrame to long format for seaborn plotting
gender_sport_distribution_melted = gender_sport_distribution_sorted.melt(id_vars=['discipline', 'Total'], var_name='gender', value_name='Count')

# Plot the data using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(x='discipline', y='Count', hue='gender', data=gender_sport_distribution_melted, palette='coolwarm')

# Rotate the x-axis labels for readability
plt.xticks(rotation=90)

# Add labels and title
plt.title('Gender Dominance in Different Sports Disciplines (Sorted by Total Count)', fontsize=16)
plt.xlabel('discipline', fontsize=12)
plt.ylabel('Number of Athletes', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[64], line 18
         15 gender_sport_distribution = gender_sport_distribution.reset_index()
         17 # Calculate total counts for sorting purposes
    ---> 18 gender_sport_distribution['Total'] = gender_sport_distribution.sum(axis=1)
         20 # Sort by total count in ascending order
         21 gender_sport_distribution_sorted = gender_sport_distribution.sort_values(by='Total', ascending=True)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:11670, in DataFrame.sum(self, axis, skipna, numeric_only, min_count, **kwargs)
      11661 @doc(make_doc("sum", ndim=2))
      11662 def sum(
      11663     self,
       (...)
      11668     **kwargs,
      11669 ):
    > 11670     result = super().sum(axis, skipna, numeric_only, min_count, **kwargs)
      11671     return result.__finalize__(self, method="sum")
    

    File ~\anaconda3\Lib\site-packages\pandas\core\generic.py:12506, in NDFrame.sum(self, axis, skipna, numeric_only, min_count, **kwargs)
      12498 def sum(
      12499     self,
      12500     axis: Axis | None = 0,
       (...)
      12504     **kwargs,
      12505 ):
    > 12506     return self._min_count_stat_function(
      12507         "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
      12508     )
    

    File ~\anaconda3\Lib\site-packages\pandas\core\generic.py:12489, in NDFrame._min_count_stat_function(self, name, func, axis, skipna, numeric_only, min_count, **kwargs)
      12486 elif axis is lib.no_default:
      12487     axis = 0
    > 12489 return self._reduce(
      12490     func,
      12491     name=name,
      12492     axis=axis,
      12493     skipna=skipna,
      12494     numeric_only=numeric_only,
      12495     min_count=min_count,
      12496 )
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:11562, in DataFrame._reduce(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)
      11558     df = df.T
      11560 # After possibly _get_data and transposing, we are now in the
      11561 #  simple case where we can use BlockManager.reduce
    > 11562 res = df._mgr.reduce(blk_func)
      11563 out = df._constructor_from_mgr(res, axes=res.axes).iloc[0]
      11564 if out_dtype is not None and out.dtype != "boolean":
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1500, in BlockManager.reduce(self, func)
       1498 res_blocks: list[Block] = []
       1499 for blk in self.blocks:
    -> 1500     nbs = blk.reduce(func)
       1501     res_blocks.extend(nbs)
       1503 index = Index([None])  # placeholder
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\blocks.py:404, in Block.reduce(self, func)
        398 @final
        399 def reduce(self, func) -> list[Block]:
        400     # We will apply the function and reshape the result into a single-row
        401     #  Block with the same mgr_locs; squeezing will be done at a higher level
        402     assert self.ndim == 2
    --> 404     result = func(self.values)
        406     if self.values.ndim == 1:
        407         res_values = result
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:11481, in DataFrame._reduce.<locals>.blk_func(values, axis)
      11479         return np.array([result])
      11480 else:
    > 11481     return op(values, axis=axis, skipna=skipna, **kwds)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\nanops.py:85, in disallow.__call__.<locals>._f(*args, **kwargs)
         81     raise TypeError(
         82         f"reduction operation '{f_name}' not allowed for this dtype"
         83     )
         84 try:
    ---> 85     return f(*args, **kwargs)
         86 except ValueError as e:
         87     # we want to transform an object array
         88     # ValueError message to the more typical TypeError
         89     # e.g. this is normally a disallowed function on
         90     # object arrays that contain strings
         91     if is_object_dtype(args[0]):
    

    File ~\anaconda3\Lib\site-packages\pandas\core\nanops.py:404, in _datetimelike_compat.<locals>.new_func(values, axis, skipna, mask, **kwargs)
        401 if datetimelike and mask is None:
        402     mask = isna(values)
    --> 404 result = func(values, axis=axis, skipna=skipna, mask=mask, **kwargs)
        406 if datetimelike:
        407     result = _wrap_results(result, orig_values.dtype, fill_value=iNaT)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\nanops.py:477, in maybe_operate_rowwise.<locals>.newfunc(values, axis, **kwargs)
        474         results = [func(x, **kwargs) for x in arrs]
        475     return np.array(results)
    --> 477 return func(values, axis=axis, **kwargs)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\nanops.py:646, in nansum(values, axis, skipna, min_count, mask)
        643 elif dtype.kind == "m":
        644     dtype_sum = np.dtype(np.float64)
    --> 646 the_sum = values.sum(axis, dtype=dtype_sum)
        647 the_sum = _maybe_null_out(the_sum, axis, mask, values.shape, min_count=min_count)
        649 return the_sum
    

    File ~\anaconda3\Lib\site-packages\numpy\core\_methods.py:49, in _sum(a, axis, dtype, out, keepdims, initial, where)
         47 def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         48          initial=_NoValue, where=True):
    ---> 49     return umr_sum(a, axis, dtype, out, keepdims, initial, where)
    

    TypeError: can only concatenate str (not "int") to str



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected file path with double backslashes (use your file path)
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Create a pivot table to analyze gender distribution by sport
gender_sport_distribution = df.pivot_table(index='discipline', columns='gender', aggfunc='size', fill_value=0)

# Add a column for total athletes (to sort by)
gender_sport_distribution['Total'] = gender_sport_distribution.sum(axis=1)

# Sort the data by total number of athletes in ascending order
gender_sport_distribution = gender_sport_distribution.sort_values('Total', ascending=True)

# Reset the index for easier plotting
gender_sport_distribution = gender_sport_distribution.reset_index()

# Melt the DataFrame to long format for seaborn plotting
gender_sport_distribution_melted = gender_sport_distribution.melt(id_vars=['discipline', 'Total'], var_name='gender', value_name='Count')

# Plot the data using seaborn, sorted by the total number of athletes
plt.figure(figsize=(12, 8))
sns.barplot(x='discipline', y='Count', hue='gender', data=gender_sport_distribution_melted, palette='coolwarm', order=gender_sport_distribution['discipline'])

# Rotate the x-axis labels for readability
plt.xticks(rotation=90)

# Add labels and title
plt.title('Gender Dominance in Different Sports Disciplines (Sorted by Total Athletes)', fontsize=16)
plt.xlabel('Discipline', fontsize=12)
plt.ylabel('Number of Athletes', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

```


    
![png](output_10_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Corrected file path (replace with your actual file path)
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Ensure 'medal_date' is in datetime format
df['medal_date'] = pd.to_datetime(df['medal_date'])

# Extract the year from the 'medal_date'
df['year'] = df['medal_date'].dt.year

# Drop rows with missing gender or year information
df_cleaned = df.dropna(subset=['gender', 'year'])

# Group the data by year and gender, counting the number of medals won by each gender
gender_year_distribution = df_cleaned.groupby(['year', 'gender']).size().unstack().fillna(0)

# Add a total medals column for each year
gender_year_distribution['Total'] = gender_year_distribution['Female'] + gender_year_distribution['Male']

# Calculate the percentage of medals won by each gender per year
gender_year_distribution['Female %'] = (gender_year_distribution['Female'] / gender_year_distribution['Total']) * 100
gender_year_distribution['Male %'] = (gender_year_distribution['Male'] / gender_year_distribution['Total']) * 100

# Plot the trend of gender disparity over time
plt.figure(figsize=(12, 8))
plt.plot(gender_year_distribution.index, gender_year_distribution['Female %'], label='Female %', color='blue', marker='o')
plt.plot(gender_year_distribution.index, gender_year_distribution['Male %'], label='Male %', color='red', marker='o')

# Add labels, title, and legend
plt.title('Gender Disparity in Medal-Winning Athletes Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Medals Won (%)', fontsize=12)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_11_0.png)
    



```python
import matplotlib.pyplot as plt

# Plot a stacked area chart
plt.figure(figsize=(12, 8))
plt.stackplot(gender_year_distribution.index, 
              gender_year_distribution['Female %'], gender_year_distribution['Male %'], 
              labels=['Female %', 'Male %'], colors=['blue', 'red'])

# Add labels, title, and legend
plt.title('Gender Disparity in Medal-Winning Athletes Over Time (Stacked Area)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Medals Won (%)', fontsize=12)
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_12_0.png)
    



```python
# Plot the number of medals won by gender for each year (grouped bar plot)
gender_year_counts = df_cleaned.groupby(['year', 'gender']).size().unstack().fillna(0)

plt.figure(figsize=(12, 8))
gender_year_counts.plot(kind='bar', stacked=False, color=['blue', 'red'], figsize=(12, 8))

# Add labels, title, and legend
plt.title('Number of Medals Won by Gender Over Time (Grouped Bar Plot)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Medals Won', fontsize=12)
plt.legend(title='Gender')

# Show the plot
plt.tight_layout()
plt.show()

```


    <Figure size 1200x800 with 0 Axes>



    
![png](output_13_1.png)
    



```python
import seaborn as sns

# Create a heatmap of gender percentage over time
plt.figure(figsize=(12, 8))
sns.heatmap(gender_year_distribution[['Female %', 'Male %']].T, cmap='coolwarm', annot=True, cbar=True, fmt='.1f')

# Add labels and title
plt.title('Heatmap of Gender Disparity in Medals Over Time', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Gender', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_14_0.png)
    



```python
# Line plot with confidence interval using seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x=gender_year_distribution.index, y='Female %', data=gender_year_distribution, label='Female %', ci='sd')
sns.lineplot(x=gender_year_distribution.index, y='Male %', data=gender_year_distribution, label='Male %', ci='sd')

# Add labels and title
plt.title('Gender Disparity in Medals Over Time with Confidence Interval', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Percentage of Medals Won (%)', fontsize=12)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

```

    C:\Users\agaje\AppData\Local\Temp\ipykernel_10948\1518403106.py:3: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar='sd'` for the same effect.
    
      sns.lineplot(x=gender_year_distribution.index, y='Female %', data=gender_year_distribution, label='Female %', ci='sd')
    C:\Users\agaje\AppData\Local\Temp\ipykernel_10948\1518403106.py:4: FutureWarning: 
    
    The `ci` parameter is deprecated. Use `errorbar='sd'` for the same effect.
    
      sns.lineplot(x=gender_year_distribution.index, y='Male %', data=gender_year_distribution, label='Male %', ci='sd')
    


    
![png](output_15_1.png)
    



```python
import pandas as pd

# Corrected file path (replace with your actual file path)
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing country or medal type information
df_cleaned = df.dropna(subset=['country', 'medal_type'])

# Group by country and medal type to count the number of medals of each type
medals_by_country = df_cleaned.groupby(['country', 'medal_type']).size().unstack().fillna(0)

# Add a total medals column for each country
medals_by_country['Total Medals'] = medals_by_country.sum(axis=1)

# Calculate the ratio of gold medals to total medals for each country
medals_by_country['Gold to Total Ratio'] = medals_by_country['Gold Medal'] / medals_by_country['Total Medals']

# Display the result, sorted by the ratio of gold to total medals
medals_by_country_sorted = medals_by_country.sort_values(by='Gold to Total Ratio', ascending=False)

# Display the countries with their gold to total medal ratios
print(medals_by_country_sorted[['Gold Medal', 'Total Medals', 'Gold to Total Ratio']])

```

    medal_type  Gold Medal  Total Medals  Gold to Total Ratio
    country                                                  
    Dominica           1.0           1.0             1.000000
    Pakistan           1.0           1.0             1.000000
    Norway            18.0          23.0             0.782609
    Algeria            2.0           3.0             0.666667
    Slovenia           2.0           3.0             0.666667
    ...                ...           ...                  ...
    Kyrgyzstan         0.0           6.0             0.000000
    Lithuania          0.0           7.0             0.000000
    Malaysia           0.0           3.0             0.000000
    Mexico             0.0           8.0             0.000000
    Zambia             0.0           1.0             0.000000
    
    [92 rows x 3 columns]
    


```python
# Add labels and title
plt.title('Gold to Total Medals Ratio by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Gold to Total Medals Ratio', fontsize=12)
plt.xticks(rotation=90)

# Display the plot
plt.tight_layout()
plt.show()
```


    
![png](output_17_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

# Corrected file path (replace with your actual file path)
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing country or medal type information
df_cleaned = df.dropna(subset=['country', 'medal_type'])

# Group by country and medal type to count the number of medals of each type
medals_by_country = df_cleaned.groupby(['country', 'medal_type']).size().unstack().fillna(0)

# Add a total medals column for each country
medals_by_country['Total Medals'] = medals_by_country.sum(axis=1)

# Calculate the ratio of gold medals to total medals for each country
medals_by_country['Gold to Total Ratio'] = medals_by_country['Gold Medal'] / medals_by_country['Total Medals']

# Sort by the ratio of gold to total medals
medals_by_country_sorted = medals_by_country.sort_values(by='Gold to Total Ratio', ascending=False)

# Plot the ratio of gold to total medals for each country
plt.figure(figsize=(12, 8))
plt.bar(medals_by_country_sorted.index, medals_by_country_sorted['Gold to Total Ratio'], color='gold')

# Add labels and title
plt.title('Gold to Total Medals Ratio by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Gold to Total Medals Ratio', fontsize=12)
plt.xticks(rotation=90)

# Display the plot
plt.tight_layout()
plt.show()

```


    
![png](output_18_0.png)
    



```python
import pandas as pd

# Corrected file path (replace with your actual file path)
file_path = "C:\\Users\\agaje\\Desktop\\EDA\\medallists.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Drop rows with missing country, discipline, or medal type information
df_cleaned = df.dropna(subset=['country', 'discipline', 'medal_type'])

# Group by country and sport (discipline), and count the number of medals won in each sport
country_sport_medals = df_cleaned.groupby(['country', 'discipline']).size().reset_index(name='Medal Count')

# Sort by country and medal count to find countries that consistently win medals in the same sport
consistent_countries = country_sport_medals[country_sport_medals['Medal Count'] > 1].sort_values(by='Medal Count', ascending=False)

# Display the top countries that consistently win medals in certain sports
print(consistent_countries)

```

               country discipline  Medal Count
    476  United States   Swimming           70
    454  United States  Athletics           58
    31       Australia   Swimming           51
    208  Great Britain  Athletics           39
    100          China   Swimming           36
    ..             ...        ...          ...
    186        Georgia  Wrestling            2
    181         France  Triathlon            2
    180         France  Taekwondo            2
    177         France    Surfing            2
    488     Uzbekistan  Wrestling            2
    
    [285 rows x 3 columns]
    


```python


```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_10948\1056480833.py in ?()
          1 # Drop rows with missing athlete, discipline, or medal type information
    ----> 2 df_cleaned_athletes = df.dropna(subset=['Athlete', 'discipline', 'medal_type'])
          3 
          4 # Group by athlete and sport (discipline), and count the number of medals won in each sport
          5 athlete_sport_medals = df_cleaned_athletes.groupby(['Athlete', 'discipline']).size().reset_index(name='Medal Count')
    

    ~\anaconda3\Lib\site-packages\pandas\core\frame.py in ?(self, axis, how, thresh, subset, inplace, ignore_index)
       6666             ax = self._get_axis(agg_axis)
       6667             indices = ax.get_indexer_for(subset)
       6668             check = indices == -1
       6669             if check.any():
    -> 6670                 raise KeyError(np.array(subset)[check].tolist())
       6671             agg_obj = self.take(indices, axis=agg_axis)
       6672 
       6673         if thresh is not lib.no_default:
    

    KeyError: ['Athlete']



```python

```
