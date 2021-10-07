package math;

public class Matrix {
    double[][] matrix;
    public Matrix(final int rows, final int cols) {
        this.matrix = new double[rows][cols];
    }

    public Matrix(double[][] source) {
        matrix = new double[source.length][source[0].length];
        for (int r = 0; r < getRows(); r++) {
            for (int c = 0; c < getCols(); c++) {
                this.set(r, c, source[r][c]);
            }
        }
    }

    public void set(int r, int c, double v) {
        matrix[r][c] = v;
    }

    public Matrix getCol(int col) {
        if (col > getCols()) {
            throw new IllegalArgumentException("Col " + col + "doesn't exist");
        }
        final double[][] newMatrix = new double[getRows()][1];

        for (int row = 0; row < getRows(); row++) {
            newMatrix[row][0] = this.matrix[row][col];
        }

        return new Matrix(newMatrix);
    }

    public int getCols() {
        if (matrix.length == 0) {
            return 0;
        }
        return matrix[0].length;
    }

    public int getRows() {
        return matrix.length;
    }

    public double get(int r, int c) {
        return matrix[r][c];
    }
    public double[] toArray() {
        double[] array = new double[getRows() * getCols()];

        for (int r = 0; r < getRows(); r++) {
            if (getCols() >= 0) System.arraycopy(matrix[r], 0, array, r, getCols());
        }
        return array;
    }

    public double dotProduct(Matrix b) {
        double[] aArray = toArray();
        double[] bArray = b.toArray();

        double result = 0;
        for (int i = 0; i < aArray.length; i++) {
            result += aArray[i] * bArray[i];
        }

        return result;
    }

    public int size() {
        return matrix[0].length * matrix.length;
    }


    /**
     * Initialize the matrix with elements which >= lower and <= upper
     * @param lower lower bound
     * @param upper upper bound
     */
    public void randomize(double lower, double upper) {
        for (int i = 0; i < getRows(); i++) {
            for (int j = 0; j < getCols(); j++) {
                this.matrix[i][j] = (Math.random() * (upper - lower)) + lower;
            }
        }
    }

    /**
     * Multiply every element in the matrix with a
     * @param a the multiply-er
     * @return the new Matrix that holds elements
     */
    public Matrix multiply(double a) {
        final double[][] result = new double[getRows()][getCols()];
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getCols(); col++) {
                result[row][col] = get(row, col) * a;
            }
        }
        return new Matrix(result);
    }

    /**
     * Add element of the matrix with another matrix elements
     * @param m the matrix to add
     * @return new matrix
     */
    public Matrix add(Matrix m) {
        if (getRows() != m.getRows()) {
            throw new IllegalArgumentException(
                    "To add the matrices they must have the same number of rows and columns.  Matrix a has "
                            + getRows()
                            + " rows and matrix b has "
                            + m.getRows() + " rows.");
        }

        if (getCols() != getCols()) {
            throw new IllegalArgumentException(
                    "To add the matrices they must have the same number of rows and columns.  Matrix a has "
                            + getCols()
                            + " cols and matrix b has "
                            + m.getCols() + " cols.");
        }

        final double[][] result = new double[getRows()][getCols()];

        for (int resultRow = 0; resultRow < getRows(); resultRow++) {
            for (int resultCol = 0; resultCol < getCols(); resultCol++) {
                result[resultRow][resultCol] = get(resultRow, resultCol)
                        + m.get(resultRow, resultCol);
            }
        }
        return new Matrix(result);
    }

    /**
     * Add matrix element on [r,c] with value v
     * @param r row number
     * @param c col number
     * @param v value to be added
     */
    public void add(int r, int c, double v) {
        double newValue = get(r, c) + v;
        set(r, c, newValue);
    }

    /**
     * Set all elements to zero
     */
    public void clear() {
        for (int r = 0; r < getRows(); r++) {
            for (int c = 0; c < getCols(); c++) {
                set(r, c, 0);
            }
        }
    }
}
