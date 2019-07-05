import slack

if __name__ == '__main__':
    client = slack.WebClient(
        token='xoxp-688240011366-686094804288-685633124868-50b993d4e38b763a94c2d5298fa64aa9')

    response = client.chat_postMessage(
        channel='DKULFJPPV',
        blocks=[{"type": "divider"}],
    )
    assert response['ok']
