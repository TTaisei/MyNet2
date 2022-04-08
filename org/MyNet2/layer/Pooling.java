package org.MyNet2.layer;

import org.MyNet2.*;

/**
 * Class for pooling layer.
 */
public class Pooling extends Layer {
    /** row of max pooling. */
    public int poolRow;
    /** Column of max pooling. */
    public int poolCol;

    public Pooling(){
        ;
    }

    /**
     * Flatten 4 dimentional matrix to 2 dimentional matrix.
     * @param in 4 dimentional matrix.
     * @return 2 dimentional matrix.
     */
    public Matrix flatten(Matrix4d in){
        Matrix rtn = new Matrix(in.shape[0], in.shape[1]*in.shape[2]*in.shape[3]);

        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < in.shape[1]; j++){
                for (int k = 0; k < in.shape[2]; k++){
                    for (int l = 0; l < in.shape[3]; l++){
                        rtn.matrix[i][j*in.shape[1]+k*in.shape[2]+l] = in.get(i).get(j).matrix[k][l];
                    }
                }
            }
        }

        return rtn;
    }

    /**
     * Doing forward propagation.
     * @param in input matrix.
     * @return Matrix instance of output.
     */
    public Matrix4d forward(Matrix4d in){
        return in.clone();
    }

    /**
     * Doing back propagation.
     */
    @Override
    public Matrix4d back(){
        ;
    }
}