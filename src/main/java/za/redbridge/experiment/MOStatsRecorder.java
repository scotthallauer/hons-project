package za.redbridge.experiment;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.training.NEATGenome;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.labels.*;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.ui.TextAnchor;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.AbstractXYDataset;
import org.jfree.data.xy.XYDataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sim.display.ChartUtilities;
import za.redbridge.experiment.MultiObjective.MultiObjectiveEA;
import za.redbridge.experiment.MultiObjective.MultiObjectiveGenome;
import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.NEATM.NEATMNetwork;

import java.awt.*;
import java.io.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.ChartUtils;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

import static za.redbridge.experiment.Utils.getLoggingDirectory;
import static za.redbridge.experiment.Utils.saveObjectToFile;

/**
 * Class for recording stats each epoch.
 *
 * Created by Danielle and Alex on 2018/08/02.
 */
public class MOStatsRecorder extends StatsRecorder {

    private static final Logger log = LoggerFactory.getLogger(StatsRecorder.class);

    private final MultiObjectiveEA trainer;
    private final ScoreCalculator calculator;
    private final boolean HyperNEATM;

    private Genome currentBestGenome;

    private Path rootDirectory;
    private Path populationDirectory;
    private Path paretoDirectory;

    private Path performanceStatsFile;
    private Path scoreStatsFile;
    private Path sensorStatsFile;
    private Path sensorParamStatsFile;
    private String type;
    private String config;
    private String folderResume;

    public MOStatsRecorder(EvolutionaryAlgorithm trainer, ScoreCalculator calculator,String type, String config, String folderResume) {

        super(trainer, calculator,type, config,folderResume);
        this.trainer = (MultiObjectiveEA) trainer;
        this.calculator = calculator;
        this.HyperNEATM = calculator.isHyperNEATM();
        this.type = type;
        this.config = config;
        this.folderResume = folderResume;
        initFiles();

    }

    private void initFiles() {
        initDirectories();
        initStatsFiles();
    }

    private void initStatsFiles() {

        performanceStatsFile = rootDirectory.resolve("timeTaken.csv");
        initStatsFile(performanceStatsFile);

        scoreStatsFile = rootDirectory.resolve("scores.csv");
        initStatsFile(scoreStatsFile);


        sensorStatsFile = rootDirectory.resolve("sensors.csv");
        initStatsFile(sensorStatsFile);
        sensorParamStatsFile = rootDirectory.resolve("sensorsParams.csv");
        super.initStatsFileSensor(sensorParamStatsFile);


    }
    private void initDirectories() {
        if(!folderResume.equals("")){
            rootDirectory=Paths.get("results", folderResume);
        }
        else {
            rootDirectory = getLoggingDirectory(type, config);
        }

        initDirectory(rootDirectory);

        populationDirectory = rootDirectory.resolve("populations");
        initDirectory(populationDirectory);

        paretoDirectory = rootDirectory.resolve("pareto-fronts");
        initDirectory(paretoDirectory);
    }

    private static void initDirectory(Path path) {
        try {
            Files.createDirectories(path);
        } catch (IOException e) {
            log.error("Unable to create directories", e);
        }
    }



    private static void initStatsFile(Path path) {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {
            writer.write("epoch,max,min,mean,standev\n");
        } catch (IOException e) {
            log.error("Unable to initialize stats file", e);
        }
    }

    public void recordIterationStats() {
        int epoch = trainer.getIteration();

        log.info("Epoch " + epoch + " complete");

        recordStats("Time", calculator.getTimeTakenStatistics(), epoch, performanceStatsFile);

        recordStats("Score", calculator.getScoreStatistics(), epoch, scoreStatsFile);

        recordStats("Sensor", calculator.getSensorStatistics(), epoch, sensorStatsFile);

        recordStats("SensorParams", calculator.getParamSensor(), epoch, sensorParamStatsFile);

        savePopulation((NEATPopulation) trainer.getPopulation(), epoch);

        saveParetoFront(epoch);

    }

    private void savePopulation(NEATPopulation population, int epoch) {
        String filename = "epoch-" + epoch + ".ser";
        Path path = populationDirectory.resolve(filename);
        saveObjectToFile(population, path);
    }

    private void saveParetoFront(int epoch)
    {
        Path directory = paretoDirectory.resolve("epoch-"+epoch);
        initDirectory(directory);

        ArrayList<MultiObjectiveGenome> pareto = trainer.getFirstParetoFront();

        final XYSeries paretoFrontSeries = new XYSeries("pareto front");
        ArrayList<String> genomesLOG = new ArrayList<>();
        for(MultiObjectiveGenome genome: pareto)
        {
            saveParetoOptimalGenome(genome, directory);
            ArrayList<Double> scoreVector = genome.getScoreVector();
            Double score1 = scoreVector.get(0);
            Double score2 = scoreVector.get(1);
            paretoFrontSeries.add(score1,score2);
            genomesLOG.add(genome.getScore()+","+score1 +","+score2);
        }

        Path txtPath = directory.resolve("P@ret0_L0g.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(txtPath, Charset.defaultCharset())) {
            writer.write("rank,performance,sensor\n");
            for(String s: genomesLOG){
                writer.write(s+"\n");
            }

        } catch (IOException e) {
            log.error("Error writing pareto optimal network info file", e);
        }
        final XYSeriesCollection paretoFront = new XYSeriesCollection(paretoFrontSeries);

        JFreeChart chartPareto = ChartFactory.createXYLineChart(
                "Pareto Front",
                "Task Performance",
                "Sensor Complexity",
                paretoFront,
                PlotOrientation.VERTICAL,
                true, true, false);

        int width = 640;   /* Width of the image */
        int height = 480;  /* Height of the image */
        File XYChart = new File( directory.toString()+"/ParetoFront.jpeg" );


        try {
            ChartUtils.saveChartAsJPEG(XYChart, chartPareto, width, height);
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }



    private void saveParetoOptimalGenome(MultiObjectiveGenome genome, Path directory)  // serialise all the genome's from iteration N which have rank 0 (ie are in first pareto front)
    {
        NEATNetwork network = decodeGenome(genome);
        if(!HyperNEATM){
            GraphvizEngine.saveGenome((NEATGenome)genome, directory.resolve("phenome-ANN"+genome.getScore()+".dot"));
        }
        else{
            GraphvizEngine.saveGenome((NEATGenome)genome, directory.resolve("genome-CPPN"+genome.getScore()+".dot"));
            GraphvizEngine.saveNetwork(network, directory.resolve("phenome-ANN"+genome.getScore()+".dot"));
        }
        saveObjectToFile(network, directory.resolve("network"+".ser"));
    }


    private NEATNetwork decodeGenome(Genome genome)
    {
        return (NEATNetwork) trainer.getCODEC().decode(genome);
    }



}

