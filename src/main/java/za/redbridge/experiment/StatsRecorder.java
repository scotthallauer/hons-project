package za.redbridge.experiment;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.training.NEATGenome;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import za.redbridge.experiment.NEAT.NEATPopulation;


import static za.redbridge.experiment.Utils.getLoggingDirectory;
import static za.redbridge.experiment.Utils.saveObjectToFile;

/**
 * Class for recording stats each epoch.
 *
 * Created by jamie on 2014/09/28.
 */
public class StatsRecorder {

    private static final Logger log = LoggerFactory.getLogger(StatsRecorder.class);

    private final EvolutionaryAlgorithm trainer;
    private final ScoreCalculator calculator;
    private final boolean evolvingMorphology;

    private Genome currentBestGenome;

    private Path rootDirectory;
    private Path populationDirectory;
    private Path bestNetworkDirectory;

    private Path performanceStatsFile;
    private Path scoreStatsFile;
    private Path sensorStatsFile;

    public StatsRecorder(EvolutionaryAlgorithm trainer, ScoreCalculator calculator) {
        this.trainer = trainer;
        this.calculator = calculator;
        this.evolvingMorphology = calculator.isEvolvingMorphology();

        initFiles();
    }

    private void initFiles() {
        initDirectories();
        initStatsFiles();
    }

    private void initDirectories() {
        rootDirectory = getLoggingDirectory();
        initDirectory(rootDirectory);

        populationDirectory = rootDirectory.resolve("populations");
        initDirectory(populationDirectory);

        bestNetworkDirectory = rootDirectory.resolve("best networks");
        initDirectory(bestNetworkDirectory);
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

        if (evolvingMorphology) {
            sensorStatsFile = rootDirectory.resolve("sensors.csv");
            initStatsFile(sensorStatsFile);
        }
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

        recordStats(calculator.getPerformanceStatistics(), epoch, performanceStatsFile);

        recordStats(calculator.getScoreStatistics(), epoch, scoreStatsFile);

        if (evolvingMorphology) {
            recordStats(calculator.getSensorStatistics(), epoch, sensorStatsFile);
        }

        savePopulation((NEATPopulation) trainer.getPopulation(), epoch);

        // Check if new best network and save it if so
        NEATGenome newBestGenome = (NEATGenome) trainer.getBestGenome();
        if (newBestGenome != currentBestGenome) {
            saveGenome(newBestGenome, epoch);
            currentBestGenome = newBestGenome;
        }
    }

    private void savePopulation(NEATPopulation population, int epoch) {
        String filename = "epoch-" + epoch + ".ser";
        Path path = populationDirectory.resolve(filename);
        saveObjectToFile(population, path);
    }

    private void saveGenome(NEATGenome genome, int epoch) {
        Path directory = bestNetworkDirectory.resolve("epoch-" + epoch);
        initDirectory(directory);

        String txt;
        if (evolvingMorphology) {
            log.info("New best genome! Epoch: " + epoch + ", score: " + genome.getScore()
                    + ", num sensors: " + genome.getInputCount());
            txt = String.format("epoch: %d, fitness: %f, sensors: %d", epoch, genome.getScore(),
                    genome.getInputCount());
        } else {
            log.info("New best genome! Epoch: " + epoch + ", score: "  + genome.getScore());
            txt = String.format("epoch: %d, fitness: %f", epoch, genome.getScore());
        }
        Path txtPath = directory.resolve("info.txt");
        try (BufferedWriter writer = Files.newBufferedWriter(txtPath, Charset.defaultCharset())) {
            writer.write(txt);
        } catch (IOException e) {
            log.error("Error writing best network info file", e);
        }

        NEATNetwork network = decodeGenome(genome);
        saveObjectToFile(network, directory.resolve("network.ser"));

        GraphvizEngine.saveGenome(genome, directory.resolve("graph.dot"));
    }

    private void recordStats(DescriptiveStatistics stats, int epoch, Path filepath) {
        double max = stats.getMax();
        double min = stats.getMin();
        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        stats.clear();

        log.debug("Recording stats - max: " + max + ", mean: " + mean);
        saveStats(filepath, epoch, max, min, mean, sd);
    }

    private NEATNetwork decodeGenome(Genome genome) {
        return (NEATNetwork) trainer.getCODEC().decode(genome);
    }

    private static void saveStats(Path path, int epoch, double max, double min, double mean,
            double sd) {
        String line = String.format("%d, %f, %f, %f, %f\n", epoch, max, min, mean, sd);

        final OpenOption[] options = {
                StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.WRITE
        };
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path,
                Charset.defaultCharset(), options))) {
            writer.append(line);
        } catch (IOException e) {
            log.error("Failed to append to log file", e);
        }
    }

}
