import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt

def leitura_matriz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        rows, cols = map(int, lines[0].split())
        matrix = np.array([list(map(float, line.split())) for line in lines[1:rows+1]])
    return matrix

def multiplicacao_parcial(matrix_a, matrix_b, start_row, end_row, result_queue):
    partial_result = np.dot(matrix_a[start_row:end_row], matrix_b)
    result_queue.put((start_row, end_row, partial_result))

def resultado_parcial(result_queue, result_matrix, num_processes):
    for _ in range(num_processes):
        start_row, end_row, partial_result = result_queue.get()
        result_matrix[start_row:end_row] = partial_result

def multiplicacao_matriz(matrix_a, matrix_b, num_processes):
    rows_a, cols_a = matrix_a.shape
    rows_b, cols_b = matrix_b.shape

    if cols_a != rows_b:
        raise ValueError("Número de colunas da matriz A deve ser igual ao número de linhas da matriz B.")

    result_matrix = np.zeros((rows_a, cols_b))
    result_queue = multiprocessing.Queue()
    processes = []

    chunk_size = rows_a // num_processes
    for i in range(num_processes):
        start_row = i * chunk_size
        end_row = (i + 1) * chunk_size if i != num_processes - 1 else rows_a
        process = multiprocessing.Process(target=multiplicacao_parcial, args=(matrix_a, matrix_b, start_row, end_row, result_queue))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    resultado_parcial(result_queue, result_matrix, num_processes)

    return result_matrix

def graficos(times, speedup):
    variations = ['P1', 'P2', 'P3', 'P4']

    plt.figure(figsize=(10, 5))
    plt.bar(variations, times, color=['blue', 'green', 'red', 'purple'])
    plt.title('Tempo de Execução por Variação')
    plt.xlabel('Variação')
    plt.ylabel('Tempo (segundos)')
    plt.savefig('tempo_execucao.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(variations, speedup, color=['blue', 'green', 'red', 'purple'])
    plt.title('Speedup por Variação')
    plt.xlabel('Variação')
    plt.ylabel('Speedup')
    plt.savefig('speedup.png')
    plt.show()

def main():
    matrix_a = leitura_matriz('4_int.txt')
    matrix_b = leitura_matriz('4_int.txt')

    num_cores = multiprocessing.cpu_count()

    variations = {
        'P1': 1,  # Sem divisão da matriz
        'P2': num_cores,  # Número de processos igual ao número de cores
        'P3': 2 * num_cores,  # Número de processos igual ao dobro do número de cores
        'P4': num_cores // 2,  # Número de processos igual à metade do número de cores
    }

    times = []
    speedup = []

    for variation, num_processes in variations.items():
        start_time = time.time()
        result_matrix = multiplicacao_matriz(matrix_a, matrix_b, num_processes)
        end_time = time.time()
        execution_time = end_time - start_time
        times.append(execution_time)

        with open(f'resultado{variation}.txt', 'w') as file:
            file.write(f"{variation} - (Sem processos)\n" if variation == 'P1' else f"{variation} - (Com {num_processes} processos)\n")
            file.write(f"{num_cores} - (Número de cores na máquina)\n")
            file.write(f"0 - (Número de computadores remotos)\n")
            file.write(f"{result_matrix.shape[0]} {result_matrix.shape[1]} - (Tamanho da Matriz)\n")
            file.write(f"{execution_time} - (Tempo de Execução)\n\n")
            file.write("(Matriz resultante)\n")
            np.savetxt(file, result_matrix, fmt='%.6f')

    speedup = [times[0] / time for time in times]

    graficos(times, speedup)

if __name__ == '__main__':
    main()