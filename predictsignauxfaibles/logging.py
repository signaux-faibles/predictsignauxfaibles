import logging

from pymongo import monitoring


class CommandLogger(monitoring.CommandListener):
    """Class for MongoDB commands logging.

    Useful for debugging.

    """

    def started(self, event):
        logging.debug(
            f"Command {event.command_name} with request id"
            f"{event.request_id} started on server {event.connection_id}"
        )

    def succeeded(self, event):
        logging.debug(
            f"Command {event.command_name} with request id "
            f"{event.request_id} on server {event.connection_id} "
            f"succeeded in {event.duration_micros} microseconds"
        )

    def failed(self, event):
        logging.debug(
            f"Command {event.command_name} with request id "
            f"{event.request_id} on server {event.connection_id} "
            f"failed in {event.duration_micros} microseconds"
        )
