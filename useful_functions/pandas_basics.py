import pandas as pd

# Sample dataset
df = pd.DataFrame({
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Department': ['HR', 'IT', 'IT', 'HR', 'IT'],
    'Salary': [50000, 80000, 110000, 60000, 90000]
})

# Calculate group mean but keep original DataFrame size
df['Dept_Avg_Salary'] = df.groupby('Department')['Salary'].transform('mean')
print(df)
