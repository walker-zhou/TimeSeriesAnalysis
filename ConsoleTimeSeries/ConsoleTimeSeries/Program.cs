using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using System.Data.SqlClient;
using ConsoleTimeSeries;
// See https://aka.ms/new-console-template for more information

string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
string modelPath = Path.Combine(rootDir, "MLModel.zip");
var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";

MLContext mlContext = new MLContext();

DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();

string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                connectionString,
                                query);

IDataView dataView = loader.Load(dbSource);

IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", upperBound: 1);
IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Year", lowerBound: 1);

var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: "ForecastedRentals",
    inputColumnName: "TotalRentals",
    windowSize: 7,
    seriesLength: 30,
    trainSize: 365,
    horizon: 7,
    confidenceLevel: 0.95f,
    confidenceLowerBoundColumn: "LowerBoundRentals",
    confidenceUpperBoundColumn: "UpperBoundRentals");

SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);


Console.WriteLine("Hello, World!");

void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
{
    IDataView predictions = model.Transform(testData);

    IEnumerable<float> actual =
    mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
        .Select(observed => observed.TotalRentals);


}
