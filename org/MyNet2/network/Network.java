package org.MyNet2.network;

import java.io.Serializable;
import org.MyNet2.*;
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
     * @param seed Seed of random.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int seed, int inNum, Layer ... layers){
        this.setup(seed, inNum, layers);
    }

    /**
     * Constructor for this class.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int inNum, Layer ... layers){
        this.setup(0, inNum, layers);
    }

    /**
     * Construct instead of constructor.
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
                layer.setup(nextLayerInNum, layer.nodesNum, layer.afType, seed);
                nextLayerInNum = layer.nodesNum;
                break;
            case "Conv":
                nextLayerInNum = layer.kernelNum * layer.outRow * layer.outCol;
                break;
            case "MaxPooling":
                nextLayerInNum = layer.channelNum * layer.outRow * layer.outCol;
                System.out.println(nextLayerInNum);
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