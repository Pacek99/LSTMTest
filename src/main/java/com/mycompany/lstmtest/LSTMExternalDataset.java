package com.mycompany.lstmtest;

import com.mycompany.lstmtest.LabelLastTimeStepPreProcessor;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.CompositeDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTMExternalDataset {

    private static final Logger log = LoggerFactory.getLogger(LSTMExternalDataset.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseDir = new File("src/main/resources/uciLSTMExternalDataset/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static int trainCount;
    private static int testCount;

    private static List<double[]> oneActivity;
    private static String csvSplitBy = "\t";

    private static int[] poctyAktivit;
    private static int[] numerOfSamplesForTraining;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        //if we already have dataset computed and commented lines 77-78 and 91
        //trainCount = 634;
        //testCount = 264;

        trainCount = 0;
        testCount = 0;

        //set value for external dataset 70% train a 30% test
        poctyAktivit = new int[5];
        numerOfSamplesForTraining = new int[5];
        numerOfSamplesForTraining[0] = 47;
        numerOfSamplesForTraining[1] = 48;
        numerOfSamplesForTraining[2] = 243;
        numerOfSamplesForTraining[3] = 243;
        numerOfSamplesForTraining[4] = 48;


        //generate dataset from external dataset
        processDataExternalDataset("src/main/resources/Phones_accelerometer.csv");

        //LSTM neuronka odtial dalej
        // ----- Load the training data -----
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "\\%d.csv", 0, trainCount-1));
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "\\%d.csv", 0, trainCount-1));
        } catch (IOException ex) {
            ex.printStackTrace();
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            ex.printStackTrace();
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }

        int miniBatchSize = 5;
        int numLabelClasses = 5;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //Normalize the training data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);              //Collect training data statistics
        trainData.reset();

        //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
        //trainData.setPreProcessor(normalizer);
        trainData.setPreProcessor(new CompositeDataSetPreProcessor(normalizer, new LabelLastTimeStepPreProcessor()));


        // ----- Load the test data -----
        //Same process as for the training data.
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(0, csvSplitBy);
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        try {
            testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, testCount-1));
            testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, testCount-1));
        } catch (IOException ex) {
            ex.printStackTrace();
            //java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            ex.printStackTrace();;
//            java.util.logging.Logger.getLogger(ExternalDatasetVersion.class.getName()).log(Level.SEVERE, null, ex);
        }
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
            false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        //testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
        testData.setPreProcessor(new CompositeDataSetPreProcessor(normalizer, new LabelLastTimeStepPreProcessor()));

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            .weightInit(WeightInit.XAVIER)
            .updater(new Nesterovs(0.005))
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
            .gradientNormalizationThreshold(0.5)
            .seed(123)
            .graphBuilder()
            .addInputs("input")
            .addLayer("L1", new LSTM.Builder().nIn(3).nOut(5).activation(Activation.TANH).build(),"input")
            .addLayer("L2", new LSTM.Builder().nIn(5).nOut(5).activation(Activation.TANH).build(),"L1")
            .addLayer("globalPoolMax", new GlobalPoolingLayer.Builder().poolingType(PoolingType.MAX).dropOut(0.5).build(), "L2")
            .addLayer("globalPoolAvg", new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).dropOut(0.5).build(), "L2")
            .addLayer("D1", new DenseLayer.Builder().nIn(10).nOut(5).activation(Activation.TANH).build(), "globalPoolMax", "globalPoolAvg")
            .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX).nIn(5).nOut(numLabelClasses).build(), "D1")
            .setOutputs("output")
            .build();

        ComputationGraph net = new ComputationGraph(configuration);
        net.init();

        net.setListeners(new ScoreIterationListener(20));   //Print the score (loss function value) every 20 iterations


        // ----- Train the network, evaluating the test set performance at each epoch -----
        int nEpochs = 10;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);

            Evaluation evaluation = net.evaluate(testData);
            log.info(evaluation.stats());

            testData.reset();
            trainData.reset();
        }
        Evaluation evaluation = net.evaluate(testData);
        log.info(evaluation.stats());

        log.info("----- Example Complete -----");
    }

    public static void processDataExternalDataset(String csvFile){
        String line = "";
        String cvsSplitBy = ",";
        String currentActivity;
        String activityCode = "";
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            br.readLine();//preskoc prvy riadok s nazvami stlpcov
            while ((line = br.readLine()) != null) {
                String[] zaznam = line.split(cvsSplitBy);

                currentActivity = zaznam[9];

                switch(currentActivity){
                    case "stand":
                        activityCode = "0";
                        break;
                    case "walk":
                        activityCode = "1";
                        break;
                    case "stairsup":
                        activityCode = "2";
                        break;
                    case "stairsdown":
                        activityCode = "3";
                        break;
                    case "sit":
                        activityCode = "4";
                        break;
                    default:
                        //aktivita ktoru neuvazujeme
                        activityCode = "10";
                }

                if (activityCode.equals("10")) {
                    continue;
                }

                oneActivity = new ArrayList<>();
                //add values for xAxis, yAxis, zAxis, activityCode
                oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                    Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                while ((line = br.readLine()) != null) {
                    zaznam = line.split(cvsSplitBy);
                    if (zaznam[9].equals(currentActivity)) {
                        //add values for xAxis, yAxis, zAxis, activityCode
                        oneActivity.add(new double[]{Double.parseDouble(zaznam[3]), Double.parseDouble(zaznam[4]),
                            Double.parseDouble(zaznam[5]), Double.parseDouble(activityCode)});
                    } else {
                        System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                        if (poctyAktivit[Integer.parseInt(activityCode)] <= numerOfSamplesForTraining[Integer.parseInt(activityCode)]) {
                            poctyAktivit[Integer.parseInt(activityCode)]++;
                            computeExternalDataset(oneActivity, true);
                        } else {
                            computeExternalDataset(oneActivity, false);
                        }

                        oneActivity = new ArrayList<>();
                        break;
                    }
                }
                //ak sme do�li na koniec suboru a posledna aktivita vo file tak ju spracujeme
                if (oneActivity.size() != 0) {
                    System.out.println("length of activity " + currentActivity + "  :" + oneActivity.size());
                    computeExternalDataset(oneActivity, true);
                    poctyAktivit[Integer.parseInt(activityCode)]++;
                    oneActivity = new ArrayList<>();
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void computeExternalDataset(List<double[]> oneActivity, boolean forTraining) {
        String features = "";
        for (double[] entry : oneActivity) {
            features = features + entry[0] + csvSplitBy + entry[1] + csvSplitBy + entry[2] + "\n";
        }

        //Write output in a format we can read, in the appropriate locations
        File outPathFeatures;
        File outPathLabels;

        if (forTraining) {
            outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
            outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
        } else {
            outPathFeatures = new File(featuresDirTest, testCount + ".csv");
            outPathLabels = new File(labelsDirTest, testCount + ".csv");
        }

        try {
            FileUtils.writeStringToFile(outPathFeatures, features);
            FileUtils.writeStringToFile(outPathLabels, String.valueOf((int)oneActivity.get(0)[3]));
        } catch (IOException ex) {
            ex.printStackTrace();
        }

        if (forTraining) {
            trainCount++;
        } else {
            testCount++;
        }
    }
}