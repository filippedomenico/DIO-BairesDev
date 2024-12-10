DESCRIÇÃO
O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB. 
Neste projeto, você pode usar sua própria base de dados (exemplo: fotos suas, dos seus pais, dos seus amigos, dos seus animais domésticos, etc), 
o exemplo de gatos e cachorros, pode ser substituído por duas outras classes do seu interesse. O Dataset criado em nosso projeto anterior, pode ser utilizado agora.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Para este projeto, tomei a liberdade em modificar para que fossem captadas imagens diretamente da internet:

"def baixar_imagens(urls, pasta_destino):
    os.makedirs(pasta_destino, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                caminho_arquivo = os.path.join(pasta_destino, f"imagem_{i}.jpg")
                with open(caminho_arquivo, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Erro ao baixar {url}")
        except Exception as e:
            print(f"Erro ao processar {url}: {e}")"

Para o restante do código, algumas modificações serão feitas de acordo com o caminhar do curso.
            
