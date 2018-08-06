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
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
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
    private String txt;
    private NEATMNetwork nw;
    private String type;
    private String config;

    public MOStatsRecorder(EvolutionaryAlgorithm trainer, ScoreCalculator calculator,String type, String config) {
        super(trainer, calculator,type, config);
        this.trainer = (MultiObjectiveEA) trainer;
        this.calculator = calculator;
        this.HyperNEATM = calculator.isHyperNEATM();

        initFiles();
    }

    private void initFiles() {
        initDirectories();
        initStatsFiles();
    }

    private void initDirectories() {
        rootDirectory = getLoggingDirectory(type, config);
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

    private void initStatsFiles() {
        performanceStatsFile = rootDirectory.resolve("performance.csv");
        initStatsFile(performanceStatsFile);

        scoreStatsFile = rootDirectory.resolve("scores.csv");
        initStatsFile(scoreStatsFile);

        sensorStatsFile = rootDirectory.resolve("sensors.csv");
        initStatsFile(sensorStatsFile);
    }

    private static void initStatsFile(Path path) {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {
            writer.write("epoch, max, min, mean, standev\n");
        } catch (IOException e) {
            log.error("Unable to initialize stats file", e);
        }
    }

    public void recordIterationStats() {
        int epoch = trainer.getIteration();

        log.info("Epoch " + epoch + " complete");

        recordStats("Performance", calculator.getTimeTakenStatistics(), epoch, performanceStatsFile);

        recordStats("Score", calculator.getScoreStatistics(), epoch, scoreStatsFile);

        recordStats("Sensor", calculator.getSensorStatistics(), epoch, sensorStatsFile);

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

        final LabeledXYDataset paretoFront = new LabeledXYDataset(pareto.size());

        for(MultiObjectiveGenome genome: pareto)
        {
            saveParetoOptimalGenome(genome, directory);
            ArrayList<Double> scoreVector = genome.getScoreVector();
            paretoFront.add(scoreVector.get(0),scoreVector.get(1),genome.getScore()+"");

        }




        JFreeChart paretoChart = createChart(paretoFront);

        int width = 640;   /* Width of the image */
        int height = 480;  /* Height of the image */
        File XYChart = new File( directory.toString()+"/"+ "XYLineChart.jpeg" );
        try {
            ChartUtils.saveChartAsJPEG(XYChart, paretoChart, width, height);
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }

    private static JFreeChart createChart(final XYDataset dataset) {
        NumberAxis domain = new NumberAxis("Peformance");
        NumberAxis range = new NumberAxis("Sensor");
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
                "Pareto", JFreeChart.DEFAULT_TITLE_FONT, plot, false);
        return chart;
    }

    private void saveParetoOptimalGenome(MultiObjectiveGenome genome, Path directory)  // serialise all the genome's from iteration N which have rank 0 (ie are in first pareto front)
    {
        /* TODO
        Path txtPath = directory.resolve("info.txt");
        try (BufferedWriter writer = Files.newBufferedWriter(txtPath, Charset.defaultCharset())) {
            writer.write(txt);
        } catch (IOException e) {
            log.error("Error writing pareto optimal network info file", e);
        }
        */

        NEATNetwork network = decodeGenome(genome);
        saveObjectToFile(network, directory.resolve("network"+genome.getScore()+".ser"));   // a network's score is its crowding distance index within its front

        GraphvizEngine.saveGenome((NEATGenome) genome, directory.resolve("graph"+genome.getScore()+".dot"));
        GraphvizEngine.saveNetwork(network, directory.resolve("network"+genome.getScore()+".dot"));
    }


    private NEATNetwork decodeGenome(Genome genome)
    {
        return (NEATNetwork) trainer.getCODEC().decode(genome);
    }

    private class LabeledXYDataset extends AbstractXYDataset {

        private List<Number> x;
        private List<Number> y;
        private List<String> label;

        public LabeledXYDataset(int N) {
            x = new ArrayList<Number>(N);
            y = new ArrayList<Number>(N);
            label = new ArrayList<String>(N);
        }

        public void add(double x, double y, String label){
            this.x.add(x);
            this.y.add(y);
            this.label.add(label);
        }

        public String getLabel(int series, int item) {
            return label.get(item);
        }

        @Override
        public int getSeriesCount() {
            return 1;
        }

        @Override
        public Comparable getSeriesKey(int series) {
            return "Unit";
        }

        @Override
        public int getItemCount(int series) {
            return label.size();
        }

        @Override
        public Number getX(int series, int item) {
            return x.get(item);
        }

        @Override
        public Number getY(int series, int item) {
            return y.get(item);
        }
    }

    private static class LabelGenerator implements XYItemLabelGenerator {

        @Override
        public String generateLabel(XYDataset dataset, int series, int item) {
            LabeledXYDataset labelSource = (LabeledXYDataset) dataset;
            return labelSource.getLabel(series, item);
        }

    }

}

