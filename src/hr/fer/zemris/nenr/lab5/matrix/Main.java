package hr.fer.zemris.apr.lab1;

import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        zad2();
        //zad3();
        //zad4();
        //zad5();
        //zad6();
    }

    private static void zad2() throws IOException {
        Matrix A = Matrix.fromFile("in/2.zad/matrix.txt");
        Matrix b = Matrix.fromFile("in/2.zad/b.txt");
        Matrix x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LUP, true);
        evaluate(A, x, b);
    }

    private static void zad3() throws IOException {
        Matrix A = Matrix.fromFile("in/3.zad/matrix.txt");
        Matrix b = Matrix.fromFile("in/3.zad/b.txt");
        Matrix x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LUP, false);
        evaluate(A, x, b);
    }

    private static void zad4() throws IOException {
        Matrix A = Matrix.fromFile("in/4.zad/matrix.txt");
        Matrix b = Matrix.fromFile("in/4.zad/b.txt");

        System.out.println("LU");
        Matrix x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LU, false);
        evaluate(A, x, b);

        System.out.println("LUP");
        x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LUP, false);
        evaluate(A, x, b);
    }

    private static void zad5() throws IOException {
        Matrix A = Matrix.fromFile("in/5.zad/matrix.txt");
        Matrix b = Matrix.fromFile("in/5.zad/b.txt");
        Matrix x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LUP, false);
        evaluate(A, x, b);
    }

    private static void zad6() throws IOException {
        Matrix A = Matrix.fromFile("in/6.zad/matrix.txt");
        Matrix b = Matrix.fromFile("in/6.zad/b.txt");
        Matrix x = LinearSystemSolver.solve(A.clone(), b.clone(), LinearSystemSolver.DecompositionType.LUP, true);
        evaluate(A, x, b);
    }

    private static void evaluate(Matrix A, Matrix x, Matrix b) {
        String result = MatrixOperations.multiply(A, x).equals(b) ? "SUCCESSFULLY SOLVED" : "FAILED TO SOLVE";
        System.out.println(result);
    }
}
