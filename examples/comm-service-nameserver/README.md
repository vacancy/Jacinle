# Service Example with Name Server

This repository contains an example implementation of a simple server-client system using the `jacinle.comm.service` module. The system consists of:

- A **server** (`server.py`) that provides a remote procedure call (RPC) service for adding numbers.
- A **client** (`client.py`) that connects to the server and invokes the `add` function remotely.
- A **name server** for service discovery.

## Running the Name Server
To use the name server, set the following environment variable:

```sh
export JAC_SNS_HOST=YOUR_NAME_SERVER_HOST
```

To start a name server yourself, run:

```sh
jac-service-name-server
```

## Run the Service Server

To start the server, run:

```sh
python server.py
```

This will:
- Register the service under the name `my-service/add`.
- Start a socket-based communication service.

## Running the Client

To invoke the `add` function remotely, run:

```sh
python client.py
```

The client will:
- Connect to the name server to resolve `my-service/add`.
- Call the `add` function with parameters `(1, 2)`.
- Print the result (`3`).

