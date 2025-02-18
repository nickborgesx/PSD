import socket
import threading

def enviar_mensagem(cliente_socket):
    while True:
        mensagem = input()
        if mensagem:
            try:
                cliente_socket.send(mensagem.encode())
            except Exception as e:
                print(f"Erro ao enviar mensagem: {e}")
                break

def receber_mensagem(cliente_socket):
    while True:
        try:
            mensagem = cliente_socket.recv(1024).decode()
            if not mensagem:
                print("Conexão fechada pelo servidor.")
                break
            print(mensagem)
        except ConnectionResetError:
            print("Conexão com o servidor foi perdida.")
            break
        except Exception as e:
            print(f"Erro ao receber mensagem: {e}")
            break

def iniciar_cliente():
    cliente_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        cliente_socket.connect(('127.0.0.1', 5555))
    except Exception as e:
        print(f"Erro ao conectar ao servidor: {e}")
        return

    nome_cliente = input("Digite seu nome: ")
    if not nome_cliente or not nome_cliente.isalnum():
        print("Nome de usuário inválido. Use apenas letras e números.")
        cliente_socket.close()
        return

    try:
        cliente_socket.send(nome_cliente.encode())
    except Exception as e:
        print(f"Erro ao enviar nome de usuário: {e}")
        cliente_socket.close()
        return

    threading.Thread(target=enviar_mensagem, args=(cliente_socket,)).start()
    threading.Thread(target=receber_mensagem, args=(cliente_socket,)).start()

if __name__ == "__main__":
    iniciar_cliente()