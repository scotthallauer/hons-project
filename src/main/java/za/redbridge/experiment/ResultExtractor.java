package za.redbridge.experiment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import za.redbridge.experiment.NEATM.NEATMNetwork;
import za.redbridge.experiment.NEATM.NEATMPopulation;
import za.redbridge.simulator.config.SimConfig;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.logging.LogManager;

public class ResultExtractor {

    public static double MAX_FITNESS = 110.00;
    public static double MAX_NEW_SENSORS = 100.00;

    public static void main(String[] args) throws Exception {

        // Set date for logging and results output
        String date = new SimpleDateFormat("MMdd'T'HHmm").format(new Date());
        System.setProperty("log.name", "logback-"+date+".log");
        Utils.date = date;
        Logger log = LoggerFactory.getLogger(Main.class);

        System.out.print("What experiment number are you extracting? ");
        Scanner inKb = new Scanner(System.in);
        int expNum = inKb.nextInt();
        System.out.println();

        // Get run directories
        String runPrefix = "exp-" + expNum + "_run-";
        String epochPrefix = "epoch-";
        List<String> runDirs = getFoldersInDirectory("results");
        int numRuns = runDirs.size();
        int numEpochs = 250;
        ArrayList<Integer> runNums = new ArrayList<>(numRuns);
        for (int i = 0; i < runDirs.size() ; i++){
            int runNum = Integer.valueOf(runDirs.get(i).substring(runPrefix.length()));
            runNums.add(runNum);
        }
        Collections.sort(runNums);

        // Collect best networks stats
        int[] bestEpoch = new int[numRuns];
        double[] bestFitness = new double[numRuns];
        double[] bestOldSensors = new double[numRuns];
        double[] bestNewSensors = new double[numRuns];
        double[] bestOldNeural = new double[numRuns];
        double[] bestNewNeural = new double[numRuns];

        // Collect all epoch stats
        int[] allRun = new int[numRuns];
        double[][] allFitness = new double[numRuns][numEpochs];
        double[][] allOldSensors = new double[numRuns][numEpochs];
        double[][] allNewSensors = new double[numRuns][numEpochs];
        double[][] allOldNeural = new double[numRuns][numEpochs];
        double[][] allNewNeural = new double[numRuns][numEpochs];

        for(int i = 0; i < numRuns; i++){
            int currRunIdx = i;
            int currRunNum = runNums.get(i);
            allRun[currRunIdx] = currRunNum;
            System.out.println("Processing run " + currRunNum + "...");
            String runPath = "results/" + runPrefix + currRunNum + "/";
            String runResultsPath = runPath + getFoldersInDirectory(runPath).get(0) + "/";
            String bestNetworksPath = runResultsPath + "best networks/";
            String populationsPath = runResultsPath + "populations/";
            BufferedReader fitnessReader = new BufferedReader(new FileReader(runResultsPath + "scores.csv"));
            fitnessReader.readLine();

            // All epoch stats
            for(int j = 0; j < numEpochs; j++){
                int currEpochIdx = j;
                int currEpochNum = j+1;
                NEATMPopulation currPopulation = (NEATMPopulation) Utils.readObjectFromFile(populationsPath + epochPrefix + currEpochNum + ".ser");
                allFitness[currRunIdx][currEpochIdx] = Double.parseDouble(fitnessReader.readLine().split(",")[1]);
                allOldSensors[currRunIdx][currEpochIdx] = getOldSensors(currPopulation);
                allNewSensors[currRunIdx][currEpochIdx] = getNewSensors(currPopulation);
                allOldNeural[currRunIdx][currEpochIdx] = getOldNeural(currPopulation);
                allNewNeural[currRunIdx][currEpochIdx] = getNewNeural(currPopulation);
            }
            fitnessReader.close();

            // Best network stats
            List<String> bestNetworks = getFoldersInDirectory(bestNetworksPath);
            int bestEpochNum = 0;
            for(int j = 0; j < bestNetworks.size(); j++){
                int currEpochNum = Integer.valueOf(bestNetworks.get(j).substring(epochPrefix.length()));
                if(currEpochNum > bestEpochNum)
                    bestEpochNum = currEpochNum;
            }
            bestEpoch[currRunIdx] = bestEpochNum;
            bestFitness[currRunIdx] = allFitness[currRunIdx][bestEpochNum-1];
            bestOldSensors[currRunIdx] = allOldSensors[currRunIdx][bestEpochNum-1];
            bestNewSensors[currRunIdx] = allNewSensors[currRunIdx][bestEpochNum-1];
            bestOldNeural[currRunIdx] = allOldNeural[currRunIdx][bestEpochNum-1];
            bestNewNeural[currRunIdx] = allNewNeural[currRunIdx][bestEpochNum-1];
        }


        // Write best network results to output file
        System.out.println();
        System.out.println("Writing best network results...");
        String bestHeader = "run,epoch,raw-fitness,norm-fitness,old-sensors,raw-new-sensors,norm-new-sensors,old-neural,new-neural";
        PrintWriter bestWriter = new PrintWriter("results/exp-" + expNum + "_best.csv", "UTF-8");
        bestWriter.println(bestHeader);
        for(int i = 0; i < numRuns; i++){
            bestWriter.println(allRun[i] + "," + bestEpoch[i] + "," + bestFitness[i] + "," + (bestFitness[i]/MAX_FITNESS) + "," + bestOldSensors[i] + "," + bestNewSensors[i] + "," + (bestNewSensors[i]/MAX_NEW_SENSORS) + "," + bestOldNeural[i] + "," + bestNewNeural[i]);
        }
        bestWriter.close();

        // Write all epoch results to output files
        System.out.println("Writing all epoch results...");
        String line;
        String allHeader = "epoch";
        for(int i = 0; i < numRuns; i++){
            int currRunNum = runNums.get(i);
            allHeader += ",run-" + currRunNum;
        }

        writeRawResults(allHeader, allFitness, "all-raw-fitness", expNum, numRuns, numEpochs);
        writeNormalisedResults(allHeader, allFitness, "all-norm-fitness", expNum, numRuns, numEpochs, MAX_FITNESS);
        writeRawResults(allHeader, allOldSensors, "all-old-sensors", expNum, numRuns, numEpochs);
        writeRawResults(allHeader, allNewSensors, "all-raw-new-sensors", expNum, numRuns, numEpochs);
        writeNormalisedResults(allHeader, allNewSensors, "all-norm-new-sensors", expNum, numRuns, numEpochs, MAX_NEW_SENSORS);
        writeRawResults(allHeader, allOldNeural, "all-old-neural", expNum, numRuns, numEpochs);
        writeRawResults(allHeader, allNewNeural, "all-new-neural", expNum, numRuns, numEpochs);

    }

    private static void writeRawResults(String resultHeader, double[][] resultData, String fileSuffix, int expNum, int numRuns, int numEpochs)
            throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter resultWriter = new PrintWriter("results/exp-" + expNum + "_" + fileSuffix + ".csv", "UTF-8");
        resultWriter.println(resultHeader);
        for(int currEpochIdx = 0; currEpochIdx < numEpochs; currEpochIdx++){
            String line = String.valueOf(currEpochIdx+1);
            for(int currRunIdx = 0; currRunIdx < numRuns; currRunIdx++){
                line += "," + resultData[currRunIdx][currEpochIdx];
            }
            resultWriter.println(line);
        }
        resultWriter.close();
    }

    private static void writeNormalisedResults(String resultHeader, double[][] resultData, String fileSuffix, int expNum, int numRuns, int numEpochs, double normalisationFactor)
            throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter resultWriter = new PrintWriter("results/exp-" + expNum + "_" + fileSuffix + ".csv", "UTF-8");
        resultWriter.println(resultHeader);
        for(int currEpochIdx = 0; currEpochIdx < numEpochs; currEpochIdx++){
            String line = String.valueOf(currEpochIdx+1);
            for(int currRunIdx = 0; currRunIdx < numRuns; currRunIdx++){
                line += "," + (resultData[currRunIdx][currEpochIdx]/normalisationFactor);
            }
            resultWriter.println(line);
        }
        resultWriter.close();
    }

    private static double getOldSensors(NEATMPopulation population) {

        ScoreCalculator calculateScore =
                new ScoreCalculator(new SimConfig(), 5, null, false);

        NEATMNetwork network = (NEATMNetwork) population.getCODEC().decode(population.getBestGenome());

        return calculateScore.calculateScore2(network);

    }

    private static double getNewSensors(NEATMPopulation population) {

        ScoreCalculator calculateScore =
                new ScoreCalculator(new SimConfig(), 5, null, false);

        NEATMNetwork network = (NEATMNetwork) population.getCODEC().decode(population.getBestGenome());

        int numSensors = network.getSensorMorphology().getNumSensors();
        double sensorEnergyCost = 0.0;
        for (int i = 0 ; i < numSensors ; i++) {
            sensorEnergyCost += network.getSensorMorphology().getSensor(i).getEnergyCost();
        }

        return sensorEnergyCost;

    }

    private static double getOldNeural(NEATMPopulation population) {

        ScoreCalculator calculateScore =
                new ScoreCalculator(new SimConfig(), 5, null, false);

        NEATMNetwork network = (NEATMNetwork) population.getCODEC().decode(population.getBestGenome());

        return calculateScore.calculateScore3(network);

    }

    private static double getNewNeural(NEATMPopulation population) {

        ScoreCalculator calculateScore =
                new ScoreCalculator(new SimConfig(), 5, null, false);

        NEATMNetwork network = (NEATMNetwork) population.getCODEC().decode(population.getBestGenome());

        return calculateScore.getNeuralEnergyCost(network);

    }

    private static List<String> getFoldersInDirectory(String directoryPath) {
        File directory = new File(directoryPath);

        if(directory.exists()) {
            FileFilter directoryFileFilter = new FileFilter() {
                public boolean accept(File file) {
                    return file.isDirectory();
                }
            };

            File[] directoryListAsFile = directory.listFiles(directoryFileFilter);
            List<String> foldersInDirectory = new ArrayList<String>(directoryListAsFile.length);
            for (File directoryAsFile : directoryListAsFile) {
                foldersInDirectory.add(directoryAsFile.getName());
            }

            return foldersInDirectory;
        }else{
            return null;
        }
    }

}
