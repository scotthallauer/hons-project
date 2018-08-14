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
import java.nio.file.*;
import java.util.HashMap;

import za.redbridge.experiment.NEAT.NEATPopulation;
import za.redbridge.experiment.NEATM.NEATMNetwork;


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
    private final boolean HyperNEATM;

    private Genome currentBestGenome;

    private Path rootDirectory;
    private Path populationDirectory;
    private Path bestNetworkDirectory;

    private Path performanceStatsFile;
    private Path scoreStatsFile;
    private Path sensorStatsFile;
    private Path sensorParamStatsFile;
    private String txt;
    private NEATMNetwork nw;
    private String type;
    private String config;
    private String[] sensorTypes;
    private String folderResume;


    public StatsRecorder(EvolutionaryAlgorithm trainer, ScoreCalculator calculator, String type, String config, String folderResume) {
        this.trainer = trainer;
        this.calculator = calculator;
        this.evolvingMorphology = calculator.isEvolvingMorphology();
        this.HyperNEATM = calculator.isHyperNEATM();
        this.type = type;
        this.config = config;
        sensorTypes = calculator.names;
        this.folderResume=folderResume;

        initFiles();
    }

    private void initFiles() {
        initDirectories();
        initStatsFiles();
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
        performanceStatsFile = rootDirectory.resolve("timeTaken.csv");
        initStatsFile(performanceStatsFile);

        scoreStatsFile = rootDirectory.resolve("scores.csv");
        initStatsFile(scoreStatsFile);

        if (evolvingMorphology) {
            sensorStatsFile = rootDirectory.resolve("sensors.csv");
            initStatsFile(sensorStatsFile);
            sensorParamStatsFile = rootDirectory.resolve("sensorsParams.csv");
            initStatsFileSensor(sensorParamStatsFile);
        }

    }

    private static void initStatsFile(Path path) {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {
            writer.write("epoch,max,min,mean,standev\n");
        } catch (IOException e) {
            log.error("Unable to initialize stats file", e);
        }
    }
    private static void initStatsFileSensor(Path path) {
        try (BufferedWriter writer = Files.newBufferedWriter(path, Charset.defaultCharset())) {
            writer.write(",ColourProximity,,,,,,,,,,,,,,,ProximitySensor,,,,,,,,,,,,,,,LowResCamera,,,,,,,,,,,,,,,UltraSonic,,,,,,,,,,,,,,\n");
            writer.write(",FOV,,,,,Bearing,,,,,Range,,,,,FOV,,,,,Range,,,,,Bearing,,,,,FOV,,,,,Range,,,,,Bearing,,,,,FOV,,,,,Range,,,,,Bearing,,,,\n");
            writer.write("epoch,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size,max,min,mean,standev,size\n");
        } catch (IOException e) {
            log.error("Unable to initialize stats file", e);
        }
    }

    public void recordIterationStats() {
        int epoch = trainer.getIteration();
        log.info("Epoch " + epoch + " complete");

        recordStats("Time", calculator.getTimeTakenStatistics(), epoch, performanceStatsFile);

        recordStats("Score", calculator.getScoreStatistics(), epoch, scoreStatsFile);

        if (evolvingMorphology) {
            recordStats("Sensor", calculator.getSensorStatistics(), epoch, sensorStatsFile);
            recordStats("SensorParams", calculator.getParamSensor(), epoch, sensorParamStatsFile);
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


        NEATPopulation pop = (NEATPopulation) genome.getPopulation();
        NEATMNetwork nw = (NEATMNetwork) pop.getCODEC().decode(genome);
        double score2= calculator.calculateScore2(nw);
        if (evolvingMorphology) {
            if(HyperNEATM){

                log.info("New best genome! Epoch: " + epoch + ", task performance:: " + genome.getScore()
                        + ", sensor complexity: " + score2);
                txt = String.format("epoch: %d, fitness: %f, sensors: %f", epoch, genome.getScore(),
                        score2);
            }
            else{
                log.info("New best genome! Epoch: " + epoch + ", task performance: " + genome.getScore()
                        + ", sensor complexity: " + score2);
                txt = String.format("epoch: %d, fitness: %f, sensors: %f", epoch, genome.getScore()+0,
                        score2);
            }

        } else {
            log.info("New best genome! Epoch: " + epoch + ", task performance: "  + genome.getScore()+",score complexity:"+score2);
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
        if(!HyperNEATM){
            GraphvizEngine.saveGenome(genome, directory.resolve("phenome-ANN.dot"));
        }
        else{
            GraphvizEngine.saveGenome(genome, directory.resolve("genome-CPPN.dot"));
            GraphvizEngine.saveNetwork(network, directory.resolve("phenome-ANN.dot"));
        }



    }

    public void recordStats(String type, DescriptiveStatistics stats, int epoch, Path filepath) {
        double max = stats.getMax();
        double min = stats.getMin();
        double mean = stats.getMean();
        double sd = stats.getStandardDeviation();
        stats.clear();

        log.debug("["+type+" ]+Recording stats - max: " + max + ", mean: " + mean);
        saveStats(filepath, epoch, max, min, mean, sd);
    }

    public void recordStats(String type, HashMap<String,DescriptiveStatistics[]> stats, int epoch, Path filepath) {
        saveStatNew(filepath, epoch);

        for(int i =0;i<sensorTypes.length;i++ ){

            DescriptiveStatistics[] descriptStats = stats.get(sensorTypes[i]);
            String[] params = {"FOV","Bearing","Range"};
            for(int  j=0;j<descriptStats.length;j++){
                double max = descriptStats[j].getMax();
                double min = descriptStats[j].getMin();
                double mean = descriptStats[j].getMean();
                double sd = descriptStats[j].getStandardDeviation();
                double size = descriptStats[j].getN();
                log.debug("["+type+"-"+sensorTypes[i]+"-"+params[j] +"]+Recording stats - max: " + max + ", mean: " + mean);
                saveStat(filepath, max, min, mean, sd,size);
                descriptStats[j].clear();
            }


        }

    }

    private NEATNetwork decodeGenome(Genome genome)
    {
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

    private static void saveStat(Path path,  double max, double min, double mean,
                                  double sd, double size) {
        String line = String.format("%f, %f, %f, %f, %f,", max, min, mean, sd, size);

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

    private static void saveStatNew(Path path,  int epoch) {
        String line = String.format("\n%d,",epoch);

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
