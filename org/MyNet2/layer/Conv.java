package org.MyNet2.layer;

import java.util.ArrayList;
import org.MyNet2.*;

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

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     */
    public Conv(int channelNum, int kernelNum){
        this.channelNum = channelNum;
        this.kernelNum = kernelNum;
    }

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inRow Row of input picture.
     * @param inCol Column of input picture.
     */
    public Conv(int channelNum, int kernelNum, int inRow, int inCol){
        this.setup(channelNum, kernelNum, inRow, inCol, 0);
    }

    /**
     * Constructor for this class.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inRow Row of input picture.
     * @param inCol Column of input picture.
     * @param seed Number of seed for random class.
     */
    public Conv(int channelNum, int kernelNum, int inRow, int inCol, long seed){
        this.setup(channelNum, kernelNum, inRow, inCol, seed);
    }

    /**
     * Construct instead of constructor.
     * @param channelNum Number of channel.
     * @param kernelNum Number of kernel.
     * @param inRow Row of input picture.
     * @param inCol Column of input picture.
     * @param seed Number of seed for random class.
     */
    protected void setup(int channelNum, int kernelNum, int inRow, int inCol, long seed){
        this.channelNum = channelNum;
        this.kernelNum = kernelNum;

        int[] shape = {channelNum, kernelNum, inRow, inCol};
        this.w = new Matrix4d(shape, new Random(seed));        
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix4d instance of output.
     */
    @Override
    public Matrix4d forward(Matrix4d in){
        ;
    }
}