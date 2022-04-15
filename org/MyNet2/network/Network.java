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
                layer.setup(nextLayerInNum, layer.nodesNum, layer.afType, seed);
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