package main

import (
  "os"
  "fmt"
  "math"
  "time"
  "math/rand"
  "sort"
  "io/ioutil"
  "strings"
  "strconv"
  "net/http"
  "encoding/json"
)

type TimeSeriesData struct {
  MetaData MetaData `json:"Meta Data"`
  TimeSeries map[string]DayData `json:"Time Series (Daily)"`
}

type MetaData struct {
  Information string `json:"1. Information"`
  Symbol string `json:"2. Symbol"`
  LastRefreshed string `json:"3. Last Refreshed"`
  OutputSize string `json:"4. Output Size"`
  TimeZone string `json:"5. Time Zone"`
}

type DayData struct {
  Open string `json:"1. open"`
  High string `json:"2. high"`
  Low string `json:"3. low"`
  Close string `json:"4. close"`
  Volume string `json:"5. volume"`
}

type NeuralNetwork struct {
  inputNodes int
  hiddenNodes int
  outputNodes int
  weightsInHidden [][]float64
  weightsHiddenOut [][]float64
}

/*func relu(x float64) float64 {
  if x > 0 {
    return x
  }
  return 0
}

func drelu(x float64) float64 {
  if x > 0 {
    return 1
  }
  return 0
}*/

func leakyRelu(x float64) float64 {
  if x > 0 {
    return x
  }
  return 0.01 * x
}

func dLeakyRelu(x float64) float64 {
  if x > 0 {
    return 1
  }
  return 0.01
}

func sigmoid(x float64) float64 {
  return 1 / (1 + math.Exp(-x))
}

/*func dsigmoid(x float64) float64 {
  return x * (1 - x)
}*/

func normalizeMinMax(val,min,max float64) float64 {
  return (val - min) / (max - min)
}

func normalizeFeatures(features [][]float64) {
  for i := range features[0] {
    col := make([]float64, len(features))
    for j := range features {
      col[j] = features[j][i]
    }
    minVal := col[0]
    maxVal := col[0]
    for _, val := range col {
      if val < minVal {
        minVal = val
      }
      if val > maxVal {
        maxVal = val
      }
    }
    for j := range features {
      features[j][i] = normalizeMinMax(features[j][i], minVal, maxVal)
    }
  }
}

func iterNormalizeFeatures(features,mins,maxs []float64) {
  for i := range features {
    features[i] = (features[i] - mins[i]) / (maxs[i] - mins[i])
  }
}

func newNeuralNet(inNodes,hidNodes,outNodes int) *NeuralNetwork {
  rand.Seed(time.Now().UnixNano())
  weightsInHidden := make([][]float64, inNodes)
  for i := range weightsInHidden {
    weightsInHidden[i] = make([]float64, hidNodes)
    for j := range weightsInHidden[i] {
      weightsInHidden[i][j] = rand.Float64()
    }
  }
  weightsHiddenOut := make([][]float64, hidNodes)
  for i := range weightsHiddenOut {
    weightsHiddenOut[i] = make([]float64, outNodes)
    for j := range weightsHiddenOut[i] {
      weightsHiddenOut[i][j] = rand.Float64()
    }
  }
  return &NeuralNetwork{
    inputNodes: inNodes,
    hiddenNodes: hidNodes,
    outputNodes: outNodes,
    weightsInHidden: weightsInHidden,
    weightsHiddenOut: weightsHiddenOut,
  }
}

func (nn *NeuralNetwork) Predict(inputs []float64) []float64 {
  hiddenInputs := make([]float64, nn.hiddenNodes)
  for i := 0; i < nn.hiddenNodes; i++ {
    for j := 0; j < nn.inputNodes; j++ {
      hiddenInputs[i] += inputs[j] * nn.weightsInHidden[j][i]
    }
    hiddenInputs[i] = leakyRelu(hiddenInputs[i])
  }
  outputs := make([]float64, nn.outputNodes)
  for i := 0; i < nn.outputNodes; i++ {
    for j := 0; j < nn.hiddenNodes; j++ {
      outputs[i] += hiddenInputs[j] * nn.weightsHiddenOut[j][i]
    }
  }
  return outputs
}

func (nn *NeuralNetwork) Train(inputs,targets []float64, learningRate float64) {
  hiddenInputs := make([]float64, nn.hiddenNodes)
  // Forward Pass
  for i := 0; i < nn.hiddenNodes; i++ {
    for j := 0; j < nn.inputNodes; j++ {
      hiddenInputs[i] += inputs[j] * nn.weightsInHidden[j][i]
    }
    hiddenInputs[i] = leakyRelu(hiddenInputs[i])
  }
  outputs := make([]float64, nn.outputNodes)
  for i := 0; i < nn.outputNodes; i++ {
    for j := 0; j < nn.hiddenNodes; j++ {
      outputs[i] += hiddenInputs[j] * nn.weightsHiddenOut[j][i]
    }
  }
  // Backward Pass
  outputErrs := make([]float64, nn.outputNodes)
  for i := 0; i < nn.outputNodes; i++ {
    outputErrs[i] = targets[i] - outputs[i]
  }
  hiddenErrs := make([]float64, nn.hiddenNodes)
  for i := 0; i < nn.hiddenNodes; i++ {
    for j := 0; j < nn.outputNodes; j++ {
      hiddenErrs[i] += outputErrs[j] * nn.weightsHiddenOut[i][j]
    }
    hiddenErrs[i] *= dLeakyRelu(hiddenInputs[i])
  }
  // Update Weights
  for i := 0; i < nn.hiddenNodes; i++ {
    for j := 0; j < nn.inputNodes; j++ {
      nn.weightsInHidden[j][i] += learningRate * hiddenErrs[i] * inputs[j]
    }
  }
  for i := 0; i < nn.outputNodes; i++ {
    for j := 0; j < nn.hiddenNodes; j++ {
      nn.weightsHiddenOut[j][i] += learningRate * outputErrs[i] * hiddenInputs[i]
    }
  }
}

func main() {
  content, err := ioutil.ReadFile("key.txt")
  if err != nil {
    fmt.Println("Error reading key.txt:", err)
  }
  avKey := strings.TrimSpace(string(content))
  if avKey == "" {
    fmt.Println("AV API Key is empty or only contains whitespace.")
    return
  }
  fmt.Println(avKey)
  ticker := strings.ToUpper(strings.TrimSpace(os.Args[1]))
  apiUrl := fmt.Sprintf("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=%s", ticker, avKey)
  response, err := http.Get(apiUrl)
  if err != nil {
    fmt.Println("Error fetching AV API:", err)
    return
  }
  defer response.Body.Close()
  //fmt.Println(response.Body)
  var tSeries TimeSeriesData
  err = json.NewDecoder(response.Body).Decode(&tSeries)
  if err != nil {
    fmt.Println("Error decoding body of API response into JSON:", err)
    return
  }
  if tSeries.MetaData.Information == "" {
    fmt.Println("AV API Request Limit Exceeded or some other error")
    //return
  }
  fmt.Println("\n[AV TIME SERIES API METADATA]\n\nInformation:", tSeries.MetaData.Information)
  fmt.Println("Symbol:", tSeries.MetaData.Symbol)
  fmt.Println("Last Refreshed:", tSeries.MetaData.LastRefreshed)
  fmt.Println("Output Size:", tSeries.MetaData.OutputSize)
  fmt.Println("Time Zone:", tSeries.MetaData.TimeZone)
  var dates []string
  for d := range tSeries.TimeSeries {
    dates = append(dates, d)
  }
  sort.Strings(dates)
  /*for _, date := range dates {
    ohlcv := tSeries.TimeSeries[date]
    fmt.Println(date)
    fmt.Println(" Open:", ohlcv.Open)
    fmt.Println(" High:", ohlcv.High)
    fmt.Println(" Low:", ohlcv.Low)
    fmt.Println(" Close:", ohlcv.Close)
    fmt.Println(" Volume:", ohlcv.Volume)
    fmt.Println()
  }*/
  nn := newNeuralNet(3, 3, 1)
  var trainingIn [][]float64
  var trainingTargets [][]float64
  var mins, maxs []float64
  // Training Feature Detection
  for i := 1; i < len(dates); i++ {
    data := tSeries.TimeSeries[dates[i]]
    closePrc, _ := strconv.ParseFloat(data.Close, 64)
    prevData := tSeries.TimeSeries[dates[i-1]]
    prevClose, _ := strconv.ParseFloat(prevData.Close, 64)
    prevOpen, _ := strconv.ParseFloat(prevData.Open, 64)
    prevHigh, _ := strconv.ParseFloat(prevData.High, 64)
    prevLow, _ := strconv.ParseFloat(prevData.Low, 64)
    prevRange := prevHigh - prevLow
    prevReturn := (prevClose - prevOpen) / prevOpen * 100
    trainingIn = append(trainingIn, []float64{prevClose, prevRange, prevReturn})
    trainingTargets = append(trainingTargets, []float64{closePrc})
    if i == 1 {
      mins = append(mins, prevClose, prevRange, prevReturn)
	    maxs = append(maxs, prevClose, prevRange, prevReturn)
    } else {
      for j, val := range []float64{prevClose, prevRange, prevReturn} {
        if val < mins[j] {
          mins[j] = val
        }
	      if val > maxs[j] {
          maxs[j] = val
	      }
	    }
    }
  }
  normalizeFeatures(trainingIn)
  // Training Loop
  for epoch := 0; epoch < 200; epoch++ {
    for i := range trainingIn {
      nn.Train(trainingIn[i], trainingTargets[i], .001)
    }
  }
  // Post-Training Prediction Test
  prevData := tSeries.TimeSeries[dates[len(dates)-2]]
  prevClose, _ := strconv.ParseFloat(prevData.Close, 64)
  prevOpen, _ := strconv.ParseFloat(prevData.Open, 64)
  prevHigh, _ := strconv.ParseFloat(prevData.High, 64)
  prevLow, _ := strconv.ParseFloat(prevData.Low, 64)
  prevVol, _ := strconv.ParseFloat(prevData.Volume, 64)
  prevRange := prevHigh - prevLow
  prevReturn := (prevClose - prevOpen) / prevOpen * 100
  fmt.Printf("\n[PREDICTION TEST DATA: %s]\n > (O,H,L,C,Volume) = ($%.2f, $%.2f, $%.2f, $%.2f, %.2f)\n > prevRange = $%.2f\n > prevReturn = %.4f%%\n", dates[len(dates)-2], prevOpen, prevHigh, prevLow, prevClose, prevVol, prevRange, prevReturn)
  predictionTestData := []float64{prevClose, prevRange, prevReturn}
  iterNormalizeFeatures(predictionTestData, mins, maxs) 
  prediction := nn.Predict(predictionTestData)
  fmt.Printf("\n'prediction' struct / object:\n [%v]\n", prediction)
  predictedClose := prediction[0]
  targetClose, _ := strconv.ParseFloat(tSeries.TimeSeries[dates[len(dates)-1]].Close, 64)
  fmt.Printf("\n[PREDICTION TEST RESULT]\n => Predicted Close Price for %s: $%.2f\n => Actual Close Price: $%.2f\n => delta / error = $%f\n", dates[len(dates)-1], predictedClose, targetClose, predictedClose - targetClose)
}
