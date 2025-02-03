import socket, threading
from datetime import datetime

clientes = {}
grupos = {}
mensagens_pendentes = {}

def enviar_mensagem(destinatario, mensagem):
    if destinatario in clientes:
        try:
            clientes[destinatario].send(mensagem.encode())
        except:
            print(f"Erro ao enviar mensagem para {destinatario}.")
    else:
        if destinatario not in mensagens_pendentes:
            mensagens_pendentes[destinatario] = []
        mensagens_pendentes[destinatario].append(mensagem)

def listar_usuarios():
    return list(clientes.keys())

def listar_grupos():
    return list(grupos.keys())

def lidar_com_cliente(cliente_socket, endereco_cliente):
    try:
        nome_cliente = cliente_socket.recv(1024).decode()

        if nome_cliente in clientes:
            clientes[nome_cliente].close()

        clientes[nome_cliente] = cliente_socket
        print(f"{nome_cliente} conectado de {endereco_cliente}")

        if nome_cliente in mensagens_pendentes:
            for mensagem in mensagens_pendentes[nome_cliente]:
                cliente_socket.send(mensagem.encode())
            del mensagens_pendentes[nome_cliente]

        while True:
            comando = cliente_socket.recv(1024).decode()
            if not comando:
                break

            if comando.startswith("-msgt"):
                partes = comando.split(" ", 2)
                if len(partes) < 3:
                    cliente_socket.send("Erro: Use -msgt C/D/T MENSAGEM".encode())
                    continue

                _, tipo, mensagem = partes
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                mensagem_formatada = f"({nome_cliente}, TODOS, {timestamp}) {mensagem}"

                if tipo == "C":  # Enviar para todos conectados
                    for usuario in clientes.keys():
                        enviar_mensagem(usuario, mensagem_formatada)

                elif tipo == "D":  # Armazenar apenas para usuários offline
                    for usuario in listar_usuarios():
                        if usuario not in clientes:  # Apenas se estiver offline
                            mensagens_pendentes.setdefault(usuario, []).append(mensagem_formatada)

                elif tipo == "T":  # Enviar para todos conectados e armazenar para offline
                    for usuario in clientes.keys():
                        enviar_mensagem(usuario, mensagem_formatada)

                    for usuario in listar_usuarios():
                        if usuario not in clientes:  # Apenas usuários offline
                            mensagens_pendentes.setdefault(usuario, []).append(mensagem_formatada)

            elif comando.startswith("-msg"):
                partes = comando.split(" ", 3)
                if len(partes) < 4:
                    cliente_socket.send("Erro: Use -msg U/G DESTINO MENSAGEM".encode())
                    continue
                _, tipo, destino, mensagem = partes
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                mensagem_formatada = f"({nome_cliente}, {destino}, {timestamp}) {mensagem}"

                if tipo == "U":
                    enviar_mensagem(destino, mensagem_formatada)
                elif tipo == "G":
                    if destino in grupos:
                        for membro in grupos[destino]:
                            enviar_mensagem(membro, mensagem_formatada)
                        if destino not in mensagens_pendentes:
                            mensagens_pendentes[destino] = []
                        mensagens_pendentes[destino].append(mensagem_formatada)
                    else:
                        cliente_socket.send("Erro: Grupo não existe".encode())

            elif comando.startswith("-listarusuarios"):
                cliente_socket.send(f"Usuários: {', '.join(listar_usuarios())}".encode())

            elif comando.startswith("-criargrupo"):
                partes = comando.split(" ", 1)
                if len(partes) < 2:
                    cliente_socket.send("Erro: Use -criargrupo NOME_DO_GRUPO".encode())
                    continue
                nome_grupo = partes[1]
                if nome_grupo in grupos:
                    cliente_socket.send("Erro: Grupo já existente".encode())
                else:
                    grupos[nome_grupo] = []
                    cliente_socket.send(f"Grupo '{nome_grupo}' criado com sucesso.".encode())

            elif comando.startswith("-listargrupos"):
                if grupos:
                    cliente_socket.send(f"Grupos: {', '.join(listar_grupos())}".encode())
                else:
                    cliente_socket.send("Erro: Nenhum grupo cadastrado".encode())

            elif comando.startswith("-entrargrupo"):
                partes = comando.split(" ", 1)
                if len(partes) < 2:
                    cliente_socket.send("Erro: Use -entrargrupo NOME_DO_GRUPO".encode())
                    continue
                nome_grupo = partes[1]
                if nome_grupo in grupos:
                    if nome_cliente not in grupos[nome_grupo]:
                        grupos[nome_grupo].append(nome_cliente)
                    cliente_socket.send(f"Você entrou no grupo '{nome_grupo}'".encode())
                else:
                    cliente_socket.send("Erro: Grupo não existe".encode())

            elif comando.startswith("-sairgrupo"):
                (partes) = comando.split(" ", 1)
                if len(partes) < 2:
                    cliente_socket.send("Erro: Use -sairgrupo NOME_DO_GRUPO".encode())
                    continue
                nome_grupo = partes[1]
                if nome_grupo in grupos and nome_cliente in grupos[nome_grupo]:
                    grupos[nome_grupo].remove(nome_cliente)
                    cliente_socket.send(f"Você saiu do grupo '{nome_grupo}'".encode())
                else:
                    cliente_socket.send("Erro: Você não está no grupo ou ele não existe".encode())

            elif comando.startswith("-listausrgrupo"):
                partes = comando.split(" ", 1)
                if len(partes) < 2:
                    cliente_socket.send("Erro: Use -listausrgrupo NOME_DO_GRUPO".encode())
                    continue
                nome_grupo = partes[1]
                if nome_grupo in grupos:
                    cliente_socket.send(f"Usuários do grupo '{nome_grupo}': {', '.join(grupos[nome_grupo])}".encode())
                else:
                    cliente_socket.send("Erro: Grupo não cadastrado".encode())

    except Exception as e:
        print(f"Erro com o cliente {endereco_cliente}: {e}")
    finally:
        if nome_cliente in clientes and clientes[nome_cliente] == cliente_socket:
            del clientes[nome_cliente]
            cliente_socket.close()

def iniciar_servidor():
    servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    servidor.bind(('127.0.0.1', 5555))
    servidor.listen(5)
    print("Servidor aguardando conexões...")

    while True:
        cliente_socket, endereco_cliente = servidor.accept()
        threading.Thread(target=lidar_com_cliente, args=(cliente_socket, endereco_cliente)).start()

if __name__ == "__main__":
    iniciar_servidor()