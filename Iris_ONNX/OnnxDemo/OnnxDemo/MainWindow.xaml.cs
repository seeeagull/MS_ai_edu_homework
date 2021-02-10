using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Ink;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace OnnxIrisDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            btnReset.IsEnabled = false;
        }
        public void tbClick1(object sender, RoutedEventArgs e)
        {
            if (sepalLength.Text == "Enter the sepal length...")
                sepalLength.Text = "";
        }
        public void tbClick2(object sender, RoutedEventArgs e)
        {
            if (sepalWidth.Text == "Enter the sepal width...")
                sepalWidth.Text = "";
        }
        public void tbClick3(object sender, RoutedEventArgs e)
        {
            if (petalLength.Text == "Enter the petal length...")
                petalLength.Text = "";
        }
        public void tbClick4(object sender, RoutedEventArgs e)
        {
            if (petalWidth.Text == "Enter the petal width...")
                petalWidth.Text = "";
        }
        public void GetResult(double[] input)
        {
            float[] inputData = new float[4];
            for (int i = 0; i < 4; ++i)
                inputData[i] = (float)input[i];
            string modelPath = AppDomain.CurrentDomain.BaseDirectory + "iris.onnx";
            using (var session = new InferenceSession(modelPath))
            {
                var container = new List<NamedOnnxValue>();
                var tensor = new DenseTensor<float>(inputData, new int[] { 1, 4 });
                container.Add(NamedOnnxValue.CreateFromTensor<float>("f1x", tensor));
                var results = session.Run(container);
                var result = results.FirstOrDefault()?.AsTensor<float>()?.ToList();
                var max = result.IndexOf(result.Max());
            
            if(max==0)
                lbResult.Text = "Iris-setosa !";
            else if(max==1)
                lbResult.Text = "Iris-versicolor !";
            else
                lbResult.Text = "Iris-virginica !";
            }
        }
        public void btnEnterClick(object sender, RoutedEventArgs e)
        {
            btnReset.IsEnabled = true;
            btnEnter.IsEnabled = false;

            double[] input = new double[4];
            input[0] = Convert.ToDouble(sepalLength.Text);
            input[1] = Convert.ToDouble(sepalWidth.Text);
            input[2] = Convert.ToDouble(petalLength.Text);
            input[3] = Convert.ToDouble(petalWidth.Text);
            GetResult(input);
        }
        public void btnResetClick(object sender, RoutedEventArgs e)
        {
            sepalLength.Text = "Enter the sepal length...";
            sepalWidth.Text = "Enter the sepal width...";
            petalLength.Text = "Enter the petal length...";
            petalWidth.Text = "Enter the petal width...";
            lbResult.Text = "";
            btnEnter.IsEnabled = true;
            btnReset.IsEnabled = false;
        }
    }
}
