import pandas as pd
class OutputDataFrame:
    def __init__(
        self,
        columns: list,
        path: str ,
        add_id: bool = False
    ):
        self.columns = columns
        self.path = path

        # Ensure the directory exists
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        if add_id:
        
            self.df = pd.DataFrame(columns=["id"] + self.columns)
            self.current_index = 0
        else:
            self.df = pd.DataFrame(columns=self.columns)
            self.current_index = None

    def add_rows(self, rows: list):
        """
        Add multiple rows to the DataFrame.

        Arguments:
            rows (list): A list of dictionaries, each representing a row of data.
        """
        # before add the
        if self.current_index is not None:
            for row in rows:
                row["id"] = self.current_index
                self.current_index += 1
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.save()

    def save(self):
        """
        Save the DataFrame to a CSV file.
        """
        self.df.to_csv(self.path, index=False)
