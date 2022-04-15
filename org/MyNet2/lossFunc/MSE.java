package org.MyNet2.lossFunc;

import java.lang.Math;
import org.MyNet2.*;

/**
 * Class for loss function.
 */
public class MSE extends LossFunction {
    /**
     * Constructor for this class.
     */
    public MSE(){
        ;
    }

    /**
     * Calcurate this loss function.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return Diiference between y and b.
     */
    @Override
    public double calc(Matrix y, Matrix t){
        double rtn = 0.;

        for (int i = 0; i < y.row; i++){
            for (int j = 0; j < y.col; j++){
                rtn.matrix[i][0] += Math.pow(y.matrix[i][0] - t.matrix[i][0], 2);
            }
        }

        return rtn / y.row;
    }

    /**
     * Calcurate this loss function's differential.
     * @param y Matrix of network's output.
     * @param t Matrix of actual data.
     * @return The result of differentialting the difference between y and t.
     */
    @Override
    public Matrix diff(Matrix y, Matrix t){
        Matrix rtn = new Matrix(y.row, y.col);

        for (int i = 0; i < y.row; i++){
            for (int j = 0; j < y.col; j++){
                rtn.matrix[i][j] = (y.matrix[i][0] - t.matrix[i][0]) * 2;
            }
        }

        return rtn;
    }

    @Override
    public String toString(){
        return "MeanSquaredError";
    }
}