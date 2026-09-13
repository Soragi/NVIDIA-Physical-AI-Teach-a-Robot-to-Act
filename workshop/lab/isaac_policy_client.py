"""Bound native GR00T socket waits without changing its policy protocol."""


def install_policy_timeouts():
    import zmq
    from gr00t.policy import server_client
    if getattr(server_client.PolicyClient,'_workshop_bounded',False):
        return server_client.PolicyClient

    class BoundedPolicyClient(server_client.PolicyClient):
        _workshop_bounded=True

        def _init_socket(self):
            previous=getattr(self,'socket',None)
            if previous is not None: previous.close(linger=0)
            super()._init_socket()
            # The pinned native client stores timeout_ms but does not apply it.
            self.socket.setsockopt(zmq.RCVTIMEO,self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO,self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER,0)

    server_client.PolicyClient=BoundedPolicyClient
    return BoundedPolicyClient
