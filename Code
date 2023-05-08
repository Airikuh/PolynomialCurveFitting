import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os 
import warnings
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter('ignore')

class Project1(object):
 
 #Initialize Training and Test Files
    def __init__(self,train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file

#Standard Scaling Normalization Algorithm 
    def standard_scaling(self, values):
        new_mean = np.mean(values)
        new_std = np.std(values)
        sub = values - new_mean
        normalized = sub / new_std
        return normalized

#Parse Training and Testing Data files and Pull out Information For later Calculations
    def parse_dat_files(self):
        column_titles = ['Year', 'Population']
        #train_data = pd.read_csv(trainFile, header=None, delimiter=' ')
        #test_data = pd.read_csv(testFile, header=None, delimiter=' ')      
        #Call the Standard Scaling Normalization Algorithm and Scale the Data
        train_data = pd.read_table(self.train_file, sep = ' ', header = None, names = column_titles)
        train_normalized_year = self.standard_scaling(train_data['Year'])
        train_normalized_Pop = self.standard_scaling(train_data['Population'])
        train_data['Normalized_Year'] = train_normalized_year
        train_data['Normalized_Pop'] = train_normalized_Pop     

        test_data = pd.read_table(self.test_file, sep = ' ', header = None, names = column_titles)
       #Call the Standard Scaling Normalization Algorithm and Scale the Data
        test_normalized_year = self.standard_scaling(test_data['Year'])
        test_normalized_Pop = self.standard_scaling(test_data['Population'])
        test_data['Normalized_Year'] = test_normalized_year
        test_data['Normalized_Pop'] = test_normalized_Pop
        return train_data, test_data



#Determine the Optimal Degree via 6-fold Cross-Validation, getting data in 
#Sequential Order
    def optimal_degree(self):
        train_data,test_data = self.parse_dat_files()
        X_Train = (train_data['Normalized_Year'].values.reshape(-1,1))
        Y_Train = (train_data['Population'])
        #degrees = np.arange(13)
        degrees = range(13)
        alpha = 0
        cv_folds = 6
        k_fold = KFold(n_splits = cv_folds, shuffle = False, random_state = None)
        average_RMSE = []

        #Iterate Through Degrees 
        for d in degrees:
        #Generate the Polynomial Features and Transform 
            X_Poly = PolynomialFeatures(d).fit_transform(X_Train)
            # initialize model
            model = Ridge(alpha)
            current_MSE = []

            #Perform 6-Fold Cross-Validation using a For Loop to Iterate
            for train_idx, test_idx in k_fold.split(X_Poly):
                X_Trained, X_Test = X_Poly[train_idx], X_Poly[test_idx]
                Y_Trained, Y_Test = Y_Train[train_idx], Y_Train[test_idx]                
                #Fit the Model
                model.fit(X_Trained, Y_Trained)
                #Calculate Each MSE to be used for Average RMSE Calculation
                Y_Prediction  = model.predict(X_Test)
                current_MSE.append(mean_squared_error(Y_Test, Y_Prediction))
            #Calculate the Average RMSE for this Degree
            average_RMSE.append(np.mean(np.sqrt(current_MSE)))
            #Optimal Degree
            optimal_degree = degrees[np.argmin(average_RMSE)]

        # Print the Optimal Degree
        print(f"Optimal degree: {optimal_degree}")    

    #Print Optimal Degree and Average RMSE 
        for i, d in enumerate(degrees):
            #print('Degree: :', d)
            #print('Average RMSE: ', average_RMSE)
            #Must print in this format or it will print Degree then All RMSE's
            print(f"Degree {d}: {average_RMSE[i]}")

        return degrees, average_RMSE,optimal_degree

 
#Function for other functions to get optimal degree and do calculations with that data
    def calc_optimal_degree(self):
        train_data,test_data = self.parse_dat_files()
        X_Train = (train_data['Normalized_Year'].values.reshape(-1,1))
        Y_Train = (train_data['Population'])
        #degrees = np.arange(13)
        degrees = range(13)
        alpha = 0
        cv_folds = 6
        #Sequential Cross-Validation
        k_fold = KFold(n_splits = cv_folds, shuffle = False, random_state = None)
        average_RMSE = []
 
        #Iterate Through Degrees 
        for d in degrees:
        #Generate the Polynomial Features and Transform 
            X_Poly = PolynomialFeatures(d).fit_transform(X_Train)
            # initialize model
            model = Ridge(alpha)
#Current MSE to be used for Averaging once calculations are complete
            current_MSE = []

            #Perform 6-Fold Cross-Validation using a For Loop to Iterate
            for train_idx, test_idx in k_fold.split(X_Poly):
                X_Trained, X_Test = X_Poly[train_idx], X_Poly[test_idx]
                Y_Trained, Y_Test = Y_Train[train_idx], Y_Train[test_idx]                
                #Fit the Model
                model.fit(X_Trained, Y_Trained)
                #Calculate Each MSE to be used for Average RMSE Calculation
                Y_Prediction  = model.predict(X_Test)
                current_MSE.append(mean_squared_error(Y_Test, Y_Prediction))
            #Calculate the Average RMSE for this Degree
            average_RMSE.append(np.mean(np.sqrt(current_MSE)))
            #Optimal Degree
            optimal_degree = degrees[np.argmin(average_RMSE)]
        return degrees, average_RMSE, optimal_degree


#Do Graph separately or 6 pop ups occur 
    def show_poly_rmse(self):
        degrees, average_RMSE, optimal_degree = self.calc_optimal_degree()
        print(degrees,average_RMSE)
        fig, ax = plt.subplots()
        #'o' displays filled circles at points
        ax.plot(degrees, average_RMSE, marker = 'o')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Average RMSE')
        ax.set_title('Average RMSE vs Polynomial Degree')
        plt.show()

#Function to Get the Coefficients for the Optimal Degree Polynomial Found Above
    def coefficients(self):
        #No graph print out if missing _, _,
        _, _, d = self.calc_optimal_degree()
        train_data, test_data = self.parse_dat_files()
        X_Train = train_data['Normalized_Year'].values.reshape(-1,1)
        Y_Train = train_data['Normalized_Pop']
        alpha = 0
         #Generate the Polynomial Features and Transform        
        X_Poly = PolynomialFeatures(d).fit_transform(X_Train)
        # Train the Model 
        model = Ridge(alpha)
        #Fit the Model       
        model.fit(X_Poly, Y_Train)
        #Get Coefficient-Weights of Optimal Polynomial
        coefficient_weights = model.coef_
        print("Coefficient-Weights of Polynomial Learned on All Training Data of Degree 6: ", coefficient_weights)


#Function that gets the TRAINING RMSE of the Learned Polynomial 
    def train_rmse(self):
        train_data, test_data = self.parse_dat_files()
        X_Train = (train_data['Normalized_Year'].values.reshape(-1,1))
        Y_Train = (train_data['Population'])
        _, _, d= self.calc_optimal_degree()
        #Generate the Polynomial Features and Transform 
        X_Poly = PolynomialFeatures(d).fit_transform(X_Train)
        alpha = 0
        model = Ridge(alpha) 
        #Fit the Model        
        model.fit(X_Poly, Y_Train) 
        Y_Train_Prediction = model.predict(X_Poly)
        rmse = np.sqrt(mean_squared_error(Y_Train, Y_Train_Prediction))
        print("Train RMSE of Learned Polynomial of Degree 6: ", rmse)




#Updated function to get Testing RMSE of the Learned Polynomial
#Previously had Training data included by Accident
#When they should be kept separate
    def updated_test_rmse(self):
        train_data,test_data = self.parse_dat_files()
        X_Train = (train_data['Normalized_Year'].values.reshape(-1,1))
        Y_Train = (train_data['Population'])    
        #Get Test Data from Test File
        X_Test = (test_data['Normalized_Year'].values.reshape(-1,1))
        Y_Test = (test_data['Normalized_Pop'])
#Get optimal degree information
        #Must be done this way or does not display graph
        _, _, d= self.calc_optimal_degree()
        alpha = 0
        model = Ridge(alpha) 
        X_Test_Poly = PolynomialFeatures(d).fit_transform(X_Test)
        



        #Fit the Model        
        model.fit(X_Test_Poly, Y_Test) 
        Y_Test_Prediction = model.predict(X_Test_Poly)
        rmse = np.sqrt(mean_squared_error(Y_Test, Y_Test_Prediction))
        print("Test RMSE of Learned Polynomial of Degree 6: ", rmse)




#Function to Plot the Training Data, FInal Polynomial Curves, Learned Polynomial with Degree 6, for Range of Years 1968-2023
    def plot_years(self):
        train_data,test_data = self.parse_dat_files()
        X_Train = train_data['Normalized_Year'].values.reshape(-1, 1)
        Y_Train = train_data['Population']
       #Degree of Optimal Polynomial Found Above
        d = 6
        alpha = 0
        #Generate the Polynomial Features
        poly = PolynomialFeatures(d)
        #Fit the Polynomial       
        X_Poly = poly.fit_transform(X_Train)
        #Fit the Model
        model = Ridge(alpha)
        model.fit(X_Poly, Y_Train)
        #Range of years 1968-2023 to be used for graph information
        years_range = np.linspace(1968, 2023, 100).reshape(-1, 1)
        normalized_years_range = (years_range - train_data['Year'].mean()) / train_data['Year'].std()
        # Generate polynomial features for the range of years
        poly_range_years = poly.fit_transform(normalized_years_range)
        # Predict the population for the range of years using the fitted model
        Y_Prediction = model.predict(poly_range_years)
        # Plot the data and the fitted polynomial curve for the optimal degree found above
        fig, ax = plt.subplots()
        ax.scatter(train_data['Year'], train_data['Population'], s = 10, label = 'Training Data')
        ax.plot(years_range, Y_Prediction, label='Prediction')
        ax.set_xlabel('Year')
        ax.set_ylabel('Working-Age Population')
        ax.set_title(f'Working-Age Population over Years 1968-2023, with Optimal Degree: {d}')
        ax.legend()
        plt.show()
      



if __name__ == '__main__':
    #Change to location in your computer, will not run on my laptop unless full 
    #file path is used, FIXED!!!!!!!!!!!!!!!!!!
    #inputfile = r'C:\Users\airik\Desktop\train.dat'
    #testfile = r'C:\Users\airik\Desktop\test.dat'

    #Get Training and Testing Data
    inputfile = 'train.dat'
    testfile = 'test.dat'
    #Create Object for Regression
    model = Project1(inputfile, testfile)
    #Get Optimal Degree 
    get_optimal_degree = model.optimal_degree()
    #Display Graph 
    show_graph = model.show_poly_rmse()
    #Get coefficients for optimal degree
    get_coefficients = model.coefficients() 
    #Get RMSE Training information
    get_rmse_train = model.train_rmse()
    #Get RMSE Testing information UPDATED
    get_rmse_test = model.updated_test_rmse()
    #Show final Plot with all information
    show_final_plot = model.plot_years()



