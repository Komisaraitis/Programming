import luigi
import subprocess
import os
import datetime
from pathlib import Path


class SetupEnvironment(luigi.Task):

    def output(self):
        return luigi.LocalTarget("logs/environment_ready.txt")

    def run(self):
        Path("logs").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)

        if not os.path.exists("final_animedataset.csv"):
            raise Exception("Data file not found!")

        java_home = r"C:\Program Files\Java\jre1.8.0_441"
        if os.path.exists(java_home):
            os.environ["JAVA_HOME"] = java_home

        with open(self.output().path, "w", encoding="utf-8") as f:
            f.write(f"Environment ready: {datetime.datetime.now()}\n")


class RunHadoopAnalysis(luigi.Task):

    def requires(self):
        return SetupEnvironment()

    def output(self):
        return luigi.LocalTarget("results/hadoop_results.txt")

    def run(self):
        print("Starting Hadoop analysis...")

        jar_path = "/hadoop-2.8.2/share/hadoop/tools/lib/hadoop-streaming-2.8.2.jar"

        hadoop_commands = [
            "docker cp mapper.py 6e2a938b6083:/hadoop_lab/",
            "docker cp reducer.py 6e2a938b6083:/hadoop_lab/",
            'docker exec 6e2a938b6083 bash -c "cd /hadoop_lab && chmod +x mapper.py reducer.py"',
            'docker exec 6e2a938b6083 bash -c "cd /hadoop_lab && hadoop fs -rm -r /user/hadoop/output/adventure_results 2>/dev/null || true"',
            f'docker exec 6e2a938b6083 bash -c "cd /hadoop_lab && hadoop jar {jar_path} -files mapper.py,reducer.py -mapper \\"python3 mapper.py\\" -reducer \\"python3 reducer.py\\" -input /user/hadoop/input/final_animedataset.csv -output /user/hadoop/output/adventure_results"',
            'docker exec 6e2a938b6083 bash -c "cd /hadoop_lab && hadoop fs -cat /user/hadoop/output/adventure_results/part-00000"',
        ]

        for i, cmd in enumerate(hadoop_commands):
            print(f"Executing command {i+1}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0 and i != 3:
                print(f"Warning: {result.stderr}")

        if result.returncode == 0 and i == len(hadoop_commands) - 1:
            hadoop_content = result.stdout

            with open(self.output().path, "w", encoding="utf-8") as f:
                f.write("Hadoop MapReduce Results:\n")
                f.write("=" * 50 + "\n")
                f.write(hadoop_content)

        print("Hadoop analysis completed")


class RunSparkAnalysis(luigi.Task):

    def requires(self):
        return SetupEnvironment()

    def output(self):
        return luigi.LocalTarget("results/spark_results.txt")

    def run(self):
        print("Starting Spark analysis...")

        result = subprocess.run(
            ["python", "lab_3.2.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        if result.returncode == 0:
            print("Spark analysis completed successfully")

            if os.path.exists("spark_results.txt"):
                try:
                    with open("spark_results.txt", "r", encoding="utf-8") as f:
                        spark_content = f.read()
                except UnicodeDecodeError:
                    with open("spark_results.txt", "r", encoding="cp1251") as f:
                        spark_content = f.read()

                with open(self.output().path, "w", encoding="utf-8") as f:
                    f.write("Apache Spark Results:\n")
                    f.write("=" * 50 + "\n")
                    f.write(spark_content)

                os.remove("spark_results.txt")
        else:
            print(f"Spark error: {result.stderr}")
            raise Exception("Spark analysis failed")


class GenerateReport(luigi.Task):

    def requires(self):
        return {"hadoop": RunHadoopAnalysis(), "spark": RunSparkAnalysis()}

    def output(self):
        return luigi.LocalTarget("results/final_report.html")

    def run(self):
        print("Generating final report...")

        hadoop_results = ""
        spark_results = ""

        if os.path.exists("results/hadoop_results.txt"):
            try:
                with open("results/hadoop_results.txt", "r", encoding="utf-8") as f:
                    hadoop_results = f.read()
            except UnicodeDecodeError:
                with open("results/hadoop_results.txt", "r", encoding="cp1251") as f:
                    hadoop_results = f.read()

        if os.path.exists("results/spark_results.txt"):
            try:
                with open("results/spark_results.txt", "r", encoding="utf-8") as f:
                    spark_results = f.read()
            except UnicodeDecodeError:
                with open("results/spark_results.txt", "r", encoding="cp1251") as f:
                    spark_results = f.read()

        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Anime Analysis - Final Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .hadoop {{ border-left: 5px solid #e74c3c; }}
                .spark {{ border-left: 5px solid #3498db; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Adventure Genre Anime Analysis</h1>
                <p class="timestamp">Report generated: {datetime.datetime.now()}</p>
            </div>
            
            <div class="section hadoop">
                <h2>Hadoop MapReduce Results</h2>
                <pre>{hadoop_results}</pre>
            </div>
            
            <div class="section spark">
                <h2>Apache Spark Results</h2>
                <pre>{spark_results}</pre>
            </div>
        </body>
        </html>
        """

        with open(self.output().path, "w", encoding="utf-8") as f:
            f.write(html_report)

        print("Final report generated")


class AnimeAnalysisPipeline(luigi.WrapperTask):

    def requires(self):
        return GenerateReport()

    def run(self):
        print("Complete analysis pipeline finished successfully!")
        print("Results saved in 'results/' folder")
        print("Final report: results/final_report.html")


if __name__ == "__main__":
    luigi.build([AnimeAnalysisPipeline()], local_scheduler=True)
