package org.MyNet2;

import java.util.ArrayList;
import java.util.Random;

/**
 * Class for four dimentional matrix.
 */
public class Matrix4d {
    /** Value of this matrix. */
    public ArrayList<Matrix3d> matrix;
    /** Shape of this matrix. */
    public int[] shape;

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
     * @param shape Shape of this matrix.
     */
    public Matrix4d(int[] shape){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param num Number to fill.
     */
    public Matrix4d(int[] shape, double num){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, num));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     */
    public Matrix4d(int[] shape, Random rand){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, rand));
        }
    }

    /**
     * Constructor for this class.
     * @param shape Shape of this matrix.
     * @param rand Random instance.
     * @param min Number of min for range.
     * @param max Number of max for range.
     */
    public Matrix4d(int[] shape, Random rand, double min, double max){
        this.shape = shape.clone();
        this.matrix = new ArrayList<Matrix3d>();

        int[] shapeElement = {this.shape[1], this.shape[2], this.shape[3]};
        for (int i = 0; i < this.shape[0]; i++){
            this.matrix.add(new Matrix3d(shapeElement, rand, min, max));
        }
    }

    /**
     * Flatten 4 dimentional matrix to 2 dimentional matrix.
     * @return 2 dimentional matrix.
     */
    public Matrix flatten(){
        Matrix rtn = new Matrix(this.shape[0], this.shape[1]*this.shape[2]*this.shape[3]);

        for (int i = 0; i < rtn.row; i++){
            for (int j = 0; j < this.shape[1]; j++){
                for (int k = 0; k < this.shape[2]; k++){
                    for (int l = 0; l < this.shape[3]; l++){
                        rtn.matrix[i][j*this.shape[1]+k*this.shape[2]+l] = this.matrix.get(i).matrix.get(j).matrix[k][l];
                    }
                }
            }
        }

        return rtn;
    }

    @Override
    public Matrix4d clone(){
        Matrix4d rtn = new Matrix4d(this.shape);

        for (int i = 0; i < this.shape[0]; i++){
            rtn.matrix.set(i, this.matrix.get(i).clone());
        }

        return rtn;
    }

    @Override
    public String toString(){
        String rtn = "[";
        int j, k;

        for (int i = 0; i < this.shape[0]; i++){
            if (i == 0){
                rtn += "[";
            }else{
                rtn += "\n [";
            }
            j = 0;
            for (Matrix matrix: this.matrix.get(i).matrix){
                if (j == 0){
                    rtn += "[";
                }else{
                    rtn += "\n  [";
                }
                j++;
                k = 0;
                for (double[] ele: matrix.matrix){
                    if (k == 0){
                        rtn += "[";
                    }else{
                        rtn += "\n   [";
                    }
                    k++;
                    for (double num: ele){
                        if (num < 0){
                            rtn += String.format("%.4f ", num);
                        }else{
                            rtn += String.format(" %.4f ", num);
                        }
                    }
                    rtn += "]";
                }
                rtn += "]";
            }
            rtn += "]";
        }

        rtn += "]\n";

        return rtn;
    }
}