package org.MyNet2.layer;

import java.util.Random;
import org.MyNet2.*;
import org.MyNet2.actFunc.*;

/**
 * Class for convolution layer.
 */
public class Conv extends Layer {
    /** The list of weight for this layer */
    public Matrix4d w;
    /** Number of chanel. */
    public int channelNum;
    /** Number of kernel. */
    public int kernelNum;
    /** Row of weight matrix. */
    public int wRow;
    /** Column of weight matrix. */
    public int wCol;

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param wRow Row of weight matrix.
     * @param wCol Column of input weight matrix.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int channelNum, int kernelNum, int wRow, int wCol, AFType afType){
        this.setup(channelNum, kernelNum, wRow, wCol, 0, afType);
    }

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param wRow Row of input weight matrix.
     * @param wCol Column of input weight matrix.
     * @param seed Number of seed for random class.
     * @param afType Type of activation fucntion for this layer.
     */
    public Conv(int channelNum, int kernelNum, int wRow, int wCol, long seed, AFType afType){
        this.setup(channelNum, kernelNum, wRow, wCol, seed, afType);
    }

    /**
     * Construct instead of constructor.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param wRow Row of input weight matrix.
     * @param wCol Column of input weight matrix.
     * @param seed Number of seed for random class.
     * @param afType Type of activation fucntion for this layer.
     */
    protected void setup(int channelNum, int kernelNum, int wRow, int wCol, long seed, AFType afType){
        this.channelNum = channelNum;
        this.kernelNum = kernelNum;
        this.wRow = wRow;
        this.wCol = wCol;

        this.w = new Matrix4d(new int[]{kernelNum, channelNum, wRow, wCol}, new Random(seed));

        this.afType = afType;
        switch (afType){
        case SIGMOID:
            this.actFunc = new Sigmoid();
            break;
        case RELU:
            this.actFunc = new ReLU();
            break;
        case TANH:
            this.actFunc = new Tanh();
            break;
        case LINEAR:
            this.actFunc = new Linear();
            break;
        default:
            System.out.println("ERROR: The specified activation function is wrong");
            System.exit(-1);
        }
        this.actFuncName = this.actFunc.toString();
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix4d instance of output.
     */
    @Override
    public Matrix4d forward(Matrix4d in){
        int batchSize = in.shape[0];
        int rtnRow = in.shape[2] - this.w.shape[2] + 1;
        int rtnCol = in.shape[3] - this.w.shape[3] + 1;

        Matrix4d rtn = new Matrix4d(
            new int[]{
                batchSize,
                this.kernelNum,
                rtnRow,
                rtnCol
            }
        );
        for (int b = 0; b < batchSize; b++){
            for (int k = 0; k < this.kernelNum; k++){
                for (int i = 0; i < rtnRow; i++){
                    for (int j = 0; j < rtnCol; j++){
                        for (int c = 0; c < this.channelNum; c++){
                            for (int p = 0; p < this.wRow; p++){
                                for (int q = 0; q < this.wCol; q++){
                                    rtn.matrix.get(b).matrix.get(k).matrix[i][j] += 
                                        this.w.matrix.get(k).matrix.get(c).matrix[p][q] * in.matrix.get(b).matrix.get(c).matrix[i+p][j+q];
                                }
                            }
                        }

                        rtn.matrix.get(b).matrix.set(k, this.actFunc.calc(rtn.matrix.get(b).matrix.get(k)));
                    }
                }
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        String str = String.format(
            "----------------------------------------------------------------\n"
            + "Convolution\n"
            + "channels: %d, kernels: %d, conv size: %dx%d",
            this.channelNum, this.kernelNum, this.wRow, this.wCol
        );

        return str;
    }
}