package org.MyNet2.network;

import java.io.*;
import java.util.*;
import org.MyNet2.*;
import org.MyNet2.actFunc.*;
import org.MyNet2.layer.*;

/**
 * Class for network.
 */
public class Network implements Serializable {
    /** Information of version. */
    public final String version = Version.version;
    /** Layer array for this network. */
    public Layer[] layers;

    /**
     * Exit this program.
     * @param msg Message for printing.
     */
    protected void exit(String msg){
        System.out.println(msg);
        System.exit(-1);
    }

    /**
     * Constructor for this class.
     * Only dense layer.
     * @param seed Seed of random.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int seed, int inNum, Layer ... layers){
        this.setup(seed, inNum, layers);
    }

    /**
     * Constructor for this class.
     * Only dense layer.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int inNum, Layer ... layers){
        this.setup(0, inNum, layers);
    }

    /**
     * Construct instead of constructor.
     * Only dense layer.
     * @param seed Seed of random.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    protected void setup(int seed, int inNum, Layer ... layers){
        this.layers = layers;
        int nextLayerInNum = inNum;

        for (Layer layer: this.layers){
            switch (layer.name){
            case "Dense":
                layer.setup(
                    nextLayerInNum,
                    layer.nodesNum,
                    layer.afType,
                    seed
                );
                nextLayerInNum = layer.nodesNum;
                break;
            default:
                this.exit("layer error");
            }
        }
    }

    /**
     * Construct of this class.
     * Contain convulation and pooling layer.
     * @param channeNum Numer of channel.
     * @param inRow Row of input matrix.
     * @param inCol Column of input matrix.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int channelNum, int inRow, int inCol, Layer ... layers){
        this.setup(0, channelNum, inRow, inCol, layers);
    }

    /**
     * Construct of this class.
     * Contain convulation and pooling layer.
     * @param seed Number of seed.
     * @param channeNum Numer of channel.
     * @param inRow Row of input matrix.
     * @param inCol Column of input matrix.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int seed, int channelNum, int inRow, int inCol, Layer ... layers){
        this.setup(seed, channelNum, inRow, inCol, layers);
    }

    /**
     * Construct instead of constructor.
     * Contain convulation and pooling layer.
     * @param channeNum Numer of channel.
     * @param inRow Row of input matrix.
     * @param inCol Column of input matrix.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    protected void setup(int seed, int channelNum, int inRow, int inCol, Layer ... layers){
        this.layers = layers;
        int nextLayerChannelNum = channelNum;
        int nextLayerInRow = inRow;
        int nextLayerInCol = inCol;
        int nextLayerInNum = 0;

        for (Layer layer: this.layers){
            switch (layer.name){
            case "Dense":
                layer.setup(nextLayerInNum, layer.nodesNum, layer.afType, seed);
                nextLayerInNum = layer.nodesNum;
                break;
            case "Conv":
                layer.setup(
                    nextLayerChannelNum,
                    layer.kernelNum,
                    new int[]{nextLayerInRow, nextLayerInCol},
                    new int[]{layer.wRow, layer.wCol},
                    layer.afType,
                    seed
                );
                nextLayerChannelNum = layer.kernelNum;
                nextLayerInRow = layer.outRow;
                nextLayerInCol = layer.outCol;
                nextLayerInNum = layer.kernelNum * layer.outRow * layer.outCol;
                break;
            case "MaxPooling":
                layer.setup(
                    nextLayerChannelNum,
                    new int[]{nextLayerInRow, nextLayerInCol},
                    layer.poolSize
                );
                nextLayerChannelNum = layer.channelNum;
                nextLayerInRow = layer.outRow;
                nextLayerInCol = layer.outCol;
                nextLayerInNum = layer.channelNum * layer.outRow * layer.outCol;
                break;
            default:
                this.exit("layer error");
            }
        }
    }

    /**
     * Doing forward propagation.
     * @param in Input matrix.
     * @return Output of this network.
     */
    public Matrix forward(Matrix in){
        Matrix rtn = in.clone();

        for (Layer layer: this.layers){
            rtn = layer.forward(rtn);
        }

        return rtn;
    }

    /**
     * Save this network to one file.
     * @param name Name of save file.
     */
    public void save(String name){
        try (
            FileOutputStream fos = new FileOutputStream(name);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
        ){
            oos.writeObject(this);
            oos.flush();
        }catch (IOException e){
            this.exit("IOException");
        }
    }

    /**
     * Load a network from the file.
     * @param name Name of load file.
     */
    public static Network load(String name){
        Network loadNet = null;
        try (
            FileInputStream fis = new FileInputStream(name);
            ObjectInputStream ois = new ObjectInputStream(fis);
        ){
            loadNet = (Network)ois.readObject();
            if (Network.versionHasProblem(loadNet)){
                System.exit(-1);
            }else{
                ;
            }
        }catch (IOException e){
            System.out.println("IOException");
            System.exit(-1);
        }catch (ClassNotFoundException e){
            System.out.println("ClassNotFoundException");
            System.exit(-1);
        }

        return loadNet;
    }

    /**
     * Check that the information of version has problem.
     * If the two networks have different versions, ask do you want to continue.
     * @param loaded Loaded network.
     * @return Do loaded network has problem?
     */
    protected static boolean versionHasProblem(Network loaded){
        if (!Version.version.equals(loaded.version)){
            System.out.println("The versions of the two Network classes do not match.");
            System.out.printf("The version of loaded is %s\n", loaded.version);
            System.out.printf("The version of this is %s\n", Version.version);
            System.out.println("There is a risk of serious error.");
            System.out.print("Do you want to continue? [y/n] ");

            Scanner sc = new Scanner(System.in);
            String ans = Network.nextLine(sc);

            while (!(ans.equals("y") || ans.equals("Y")
                     || ans.equals("n") || ans.equals("N"))){
                System.out.println("Choose 'y' or 'n'.");
                System.out.print("Do you want to continue? [y/n] ");
                ans = Network.nextLine(sc);
            }

            if (ans.equals("y") || ans.equals("Y")){
                return false;
            }else{
                System.out.println("End.");
                return true;
            }
        }else{
            return false;
        }
    }

    /**
     * Read next line.
     * If read String is Null, return "".
     * @param sc Scanner instance.
     * @return Readed String instance.
     */
    private static String nextLine(Scanner sc) throws NoSuchElementException {
        String ans;
        try {
            ans = sc.nextLine();
        }catch (NoSuchElementException e){
            ans = "";
        }

        return ans;
    }

    /**
     * Print summary of this network.
     */
    public void summary(){
        System.out.println(this.toString());
    }

    @Override
    public String toString(){
        String rtn = "Network\n";

        for (Layer layer: this.layers){
            rtn += layer.toString() + "\n";
        }

        return rtn + "----------------------------------------------------------------\n";
    }
}