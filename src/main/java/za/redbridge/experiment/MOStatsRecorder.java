package za.redbridge.experiment;

import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.training.NEATGenome;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.ItemLabelAnchor;
import org.jfree.chart.labels.ItemLabelPosition;
import org.jfree.chart.labels.StandardXYToolTipGenerator;
import org.jfree.chart.labels.XYItemLabelGenerator;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.TextAnchor;
import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import za.redbridge.experiment.MultiObjective.MultiObjectiveEA;
import za.redbridge.experiment.MultiObjective.MultiObjectiveGenome;
import za.redbridge.experiment.NEAT.NEATPopulation;

import java.awt.*;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static za.redbridge.experiment.Utils.getLoggingDirectory;
import static za.redbridge.experiment.Utils.saveObjectToFile;

/**
 * Class for recording stats each epoch.
 * <p>
 * Created by Danielle and Alex on 2018/08/02.
 */
public class MOStatsRecorder extends StatsRecorder
{

    private static final Logger log = LoggerFactory.getLogger(MOStatsRecorder.class);

    private final MultiObjectiveEA trainer;
    private final ScoreCalculator calculator;
    private final boolean HyperNEATM;

    private Path rootDirectory;
    private Path finalDataDirectory;
    private Path intermedDataDirectory;


    private Path populationDirectory;

    private Path generationalStatsFile;

    private Path performanceStatsFile;
    private Path scoreStatsFile;
    private Path sensorStatsFile;
    private Path sensorParamStatsFile;


    private String type;
    private String config;
    private String folderResume;

    public MOStatsRecorder(EvolutionaryAlgorithm trainer, ScoreCalculator calculator, String type, String config, String folderResume)
    {

        super(trainer, calculator, type, config, folderResume);
        this.trainer = (MultiObjectiveEA) trainer;
        this.calculator = calculator;
        this.HyperNEATM = calculator.isHyperNEATM();
        this.type = type;
        this.config = config;
        this.folderResume = folderResume;
        initFiles();

    }

    private void initFiles()
    {
        initDirectories();
        if(!folderResume.equals("")){
            initStatsFilesResume();
        }
        else{
            initStatsFiles();
        }
    }

    private void initStatsFiles()
    {

        performanceStatsFile = rootDirectory.resolve("timeTaken.csv");
        initStatsFile(performanceStatsFile);

        scoreStatsFile = rootDirectory.resolve("scores.csv");
        initStatsFile(scoreStatsFile);

        generationalStatsFile = rootDirectory.resolve("generationalStats.csv");
        initGenerationalStatsFile(generationalStatsFile);


        sensorStatsFile = rootDirectory.resolve("sensors.csv");
        initStatsFile(sensorStatsFile);
        sensorParamStatsFile = rootDirectory.resolve("sensorsParams.csv");
        super.initStatsFileSensor(sensorParamStatsFile);


    }

    private void initStatsFilesResume()
    {

        performanceStatsFile = rootDirectory.resolve("timeTaken.csv");

        scoreStatsFile = rootDirectory.resolve("scores.csv");

        generationalStatsFile = rootDirectory.resolve("generationalStats.csv");

        sensorStatsFile = rootDirectory.resolve("sensors.csv");

        sensorParamStatsFile = rootDirectory.resolve("sensorsParams.csv");
    }

    private void initDirectories()
    {
        if (!folderResume.equals(""))
        {
            rootDirectory = Paths.get("results", folderResume);
        }
        else
        {
            rootDirectory = getLoggingDirectory(type, config);
        }

        initDirectory(rootDirectory);

        populationDirectory = rootDirectory.resolve("populations");
        initDirectory(populationDirectory);

        finalDataDirectory = rootDirectory.resolve("FinalData");
        initDirectory(finalDataDirectory);

        intermedDataDirectory = rootDirectory.resolve("IntermediateData");
        initDirectory(intermedDataDirectory);

    }

    private static void initDirectory(Path path)
    {
        try
        {
            Files.createDirectories(path);
        } catch (IOException e)
        {
            log.error("Unable to create directories", e);
        }
    }

    private static void initStatsFile(Path path)
    {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset()))
        {
            writer.write("epoch,max,min,mean,standev\n");
        } catch (IOException e)
        {
            log.error("Unable to initialize stats file", e);
        }
    }

    private static void initGenerationalStatsFile(Path path)
    {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset()))
        {
            writer.write("Generation,KP Task,KP Morph,KP Neural,AVG Task,AVG Morph,AVG Neural,Max Task,Max Morph,Max Neural,Epsilon Task,Epsilon Morph,Epsilon Neural\n");
        } catch (IOException e)
        {
            log.error("Unable to initialize stats file", e);
        }
    }

    public void recordIterationStats()
    {
        int epoch = trainer.getIteration();

        log.info("Epoch " + epoch + " complete");

        recordStats("Time", calculator.getTimeTakenStatistics(), epoch, performanceStatsFile);

        recordStats("Score", calculator.getScoreStatistics(), epoch, scoreStatsFile);

        recordStats("Sensor", calculator.getSensorStatistics(), epoch, sensorStatsFile);

        recordStats("SensorParams", calculator.getParamSensor(), epoch, sensorParamStatsFile);

        savePopulation((NEATPopulation) trainer.getPopulation(), epoch);

        saveParetoFront(epoch);

    }

    private void savePopulation(NEATPopulation population, int epoch)
    {
        String filename = "epoch-" + epoch + ".ser";
        Path path = populationDirectory.resolve(filename);
        saveObjectToFile(population, path);
    }

    private Double normaliseComplexityScore(Double Score){
        return Score/100;
    }
    private Double normaliseTaskScore(Double Score){
        return Score/110;
    }



    private void saveParetoFront(int epoch)
    {

        Path directory = (epoch != Main.Args.numGenerations ) ? intermedDataDirectory : finalDataDirectory;

        ArrayList<MultiObjectiveGenome> pareto = trainer.getFirstParetoFront();

        MultiObjectiveGenome kneePoint=null;
        MultiObjectiveGenome maxTaskPerformance=null;
        MultiObjectiveGenome epsilon=null;
        Double sumTaskScore=0.0;
        Double sumMorphComplexity =0.0;
        Double sumNeuralComplexity =0.0;
        Double minDistanceUtopia = Double.MAX_VALUE;
        Double maxParetoTaskPerformance = Double.MIN_VALUE;
        Double maxParetoEpsilonComplexity = Double.MIN_VALUE;

       // final LabeledXYDataset paretoFront = new LabeledXYDataset(pareto.size());
        ArrayList<String> genomesLOG = new ArrayList<>();

        for (MultiObjectiveGenome genome : pareto)  // individual scores and complexities on front
        {
            if(directory.toString().equals(finalDataDirectory.toString()))
            {
                saveParetoOptimalGenome(genome, directory);
            }
            ArrayList<Double> scoreVector = genome.getScoreVector();
            Double score1 = normaliseTaskScore(scoreVector.get(0));
            Double score2 = normaliseComplexityScore(scoreVector.get(1));
            sumTaskScore+=score1;
            sumMorphComplexity+=score2;
            Double distanceToUtopiaPoint = Math.sqrt(Math.pow(1-score1,2)+Math.pow(1-score2,2));
            if(distanceToUtopiaPoint<minDistanceUtopia){
                kneePoint=genome;
                minDistanceUtopia =distanceToUtopiaPoint;
            }

            if(score1>maxParetoTaskPerformance){
                maxTaskPerformance = genome;
                maxParetoTaskPerformance = score1;
            }
            //paretoFront.add(score1, score2, genome.getScore() + "");
            genomesLOG.add(Main.Args.configFile+",individual,"+genome.getScore() + "," + score1 + "," + score2);
        }


        //loop to find epsilon
        for(MultiObjectiveGenome genome: pareto){
            ArrayList<Double> scoreVector = genome.getScoreVector();
            Double score1 = normaliseTaskScore(scoreVector.get(0));
            Double score2 = normaliseComplexityScore(scoreVector.get(1));
            if(score1>maxParetoTaskPerformance-0.1){//viable candidate within epsilon
                if(score2>maxParetoEpsilonComplexity){
                    maxParetoEpsilonComplexity =score2;
                    epsilon = genome;
                }
            }

        }


        genomesLOG.add(Main.Args.configFile+",average,,"  + sumTaskScore/pareto.size() + "," + sumMorphComplexity/pareto.size());
        genomesLOG.add(Main.Args.configFile+",KP," +kneePoint.getScore()+"," + normaliseTaskScore(kneePoint.getScoreVector().get(0) )+ "," + normaliseComplexityScore(kneePoint.getScoreVector().get(1)));
        genomesLOG.add(Main.Args.configFile+",Max," +maxTaskPerformance.getScore()+"," + normaliseTaskScore(maxTaskPerformance.getScoreVector().get(0) )+ "," + normaliseComplexityScore(maxTaskPerformance.getScoreVector().get(1)));
        genomesLOG.add(Main.Args.configFile+",Epsilon," +maxTaskPerformance.getScore()+"," + normaliseTaskScore(epsilon.getScoreVector().get(0) )+ "," + normaliseComplexityScore(epsilon.getScoreVector().get(1)));



        //Generation	                                       Knee Point Task	                                              Knee Point Morph	                     Knee Point Neural	       AVG Task	                      AVG Morph	              AVG Neural	                   Max Task	                                                        Max Morph	                                      Max Neural	                Epsilon Task	                                    Epsilon Morph	                        Epsilon Neural
        String generationalDataLine = epoch+",";
        generationalDataLine+=normaliseTaskScore(kneePoint.getScoreVector().get(0))+","+normaliseComplexityScore(kneePoint.getScoreVector().get(1))+","+"-999"+",";
        generationalDataLine+=sumTaskScore/pareto.size()+","+sumMorphComplexity/pareto.size()+","+sumNeuralComplexity/pareto.size()+","+"-999"+",";
        generationalDataLine+=normaliseTaskScore(maxTaskPerformance.getScoreVector().get(0))+","+normaliseComplexityScore(maxTaskPerformance.getScoreVector().get(1))+","+"-999"+",";
        generationalDataLine+=normaliseTaskScore(epsilon.getScoreVector().get(0))+","+normaliseComplexityScore(epsilon.getScoreVector().get(1))+","+"-999";

        Path txtPath = directory.resolve("paretolog"+ epoch +".csv");
        try (BufferedWriter writer = Files.newBufferedWriter(txtPath, Charset.defaultCharset()))
        {
            writer.write("config,point,rank,task performance,sensor complexity,neural complexity\n");
            for (String s : genomesLOG)
            {
                writer.write(s + "\n");
            }

        } catch (IOException e)
        {
            log.error("Error writing pareto optimal network info file", e);
        }


        try (PrintWriter writer = new PrintWriter(new FileWriter(generationalStatsFile.toFile(), true))) {
            writer.println(generationalDataLine);
        } catch (IOException e) {
            log.error("Failed to append to log file", e);
        }


        /*
        //  <-- UNCOMMENT TO SAVE IMAGE OF PARETO FRONT -->
        JFreeChart paretoChart = createChart(paretoFront);

        int width = 640;   // Width of the image
        int height = 480;  // Height of the image
        File XYChart = new File(directory.toString() + "/ParetoFront.jpeg");


        try
        {
            ChartUtils.saveChartAsJPEG(XYChart, paretoChart, width, height);
        } catch (Exception e)
        {
            e.printStackTrace();
        }
        */
    }


    private void saveParetoOptimalGenome(MultiObjectiveGenome genome, Path directory)  // serialise all the genome's from iteration N which have rank 0 (ie are in first pareto front)
    {
        NEATNetwork network = decodeGenome(genome);
        if (!HyperNEATM)
        {
            GraphvizEngine.saveGenome((NEATGenome) genome, directory.resolve("phenome-ANN-" + genome.getScore() + ".dot"), false);
        } else
        {
            GraphvizEngine.saveGenome((NEATGenome) genome, directory.resolve("genome-CPPN-" + genome.getScore() + ".dot"), true);
            GraphvizEngine.saveNetwork(network, directory.resolve("phenome-ANN" + genome.getScore() + ".dot"));
        }
        saveObjectToFile(network, directory.resolve("network" +genome.getScore()+ ".ser"));
    }


    private NEATNetwork decodeGenome(Genome genome)
    {
        return (NEATNetwork) trainer.getCODEC().decode(genome);
    }

    private static JFreeChart createChart(final XYDataset dataset)
    {
        NumberAxis domain = new NumberAxis("Task Performance");
        NumberAxis range = new NumberAxis("Sensor Complexity");
        domain.setAutoRangeIncludesZero(true);
        XYItemRenderer renderer = new XYLineAndShapeRenderer(true, false);
        renderer.setDefaultItemLabelGenerator(new LabelGenerator());
        renderer.setDefaultItemLabelPaint(Color.orange);
        renderer.setDefaultPositiveItemLabelPosition(
                new ItemLabelPosition(ItemLabelAnchor.CENTER, TextAnchor.CENTER));
        renderer.setDefaultItemLabelFont(
                renderer.getDefaultItemLabelFont().deriveFont(14f));
        renderer.setDefaultItemLabelsVisible(true);
        renderer.setDefaultToolTipGenerator(new StandardXYToolTipGenerator());
        XYPlot plot = new XYPlot(dataset, domain, range, renderer);
        JFreeChart chart = new JFreeChart(
                "Pareto Front", JFreeChart.DEFAULT_TITLE_FONT, plot, false);
        return chart;
    }

    private class LabeledXYDataset extends AbstractXYDataset
    {
        private List<Number> x;
        private List<Number> y;
        private List<String> label;

        public LabeledXYDataset(int N)
        {
            x = new ArrayList<Number>(N);
            y = new ArrayList<Number>(N);
            label = new ArrayList<String>(N);
        }

        public void add(double x, double y, String label)
        {
            this.x.add(x);
            this.y.add(y);
            this.label.add(label);
        }

        public String getLabel(int series, int item)
        {
            return label.get(item);
        }

        @Override
        public int getSeriesCount()
        {
            return 1;
        }

        @Override
        public Comparable getSeriesKey(int series)
        {
            return "Unit";
        }

        @Override
        public int getItemCount(int series)
        {
            return label.size();
        }

        @Override
        public Number getX(int series, int item)
        {
            return x.get(item);
        }

        @Override
        public Number getY(int series, int item)
        {
            return y.get(item);
        }
    }

    private static class LabelGenerator implements XYItemLabelGenerator
    {
        @Override
        public String generateLabel(XYDataset dataset, int series, int item)
        {
            LabeledXYDataset labelSource = (LabeledXYDataset) dataset;
            return labelSource.getLabel(series, item);
        }
    }


}

