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
     * Constructor for this class.
     * Only Dense layer.
     * @param inNum Number of input.
     * @param layers Each layers.
     */
    public Network(int inNum, Layer ... layers){
        this.layers = layers;
    }

    /**
     * Constructor for this class.
     * Only CNN.
     * @param inShape Shape of input.
     * @param layers Each layers.
     */
    public Network(int[] inShape, Layer ... layers){
        if (inShape.length != 2 || inShape.length != 4){
            this.exit("inShape length is wrong.");
        }
    }

    public Matrix forward(Matrix in){
        Matrix rtn = in.clone();

        for (Layer layer: this.layers){
            rtn = layer.forward(in);
        }

        return rtn;
    }
}