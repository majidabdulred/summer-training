import gradio as gr
import demodata
from app.ner import ner
from app.zsq import zero_shot_classification
from app.summarise import summarise
from app.doc_qna import document_qna
with gr.Blocks() as demo:
    gr.Label("Natural Language Processing.")
    with gr.Tab("NER"):
        gr.HTML("<h1>Name Entity Recognition</h1><br/>"
                "<p>Named Entity Recognition (NER) is a task of information extraction that involves identifying and classifying named entities mentioned in text into predefined categories such as the names of people, organizations, locations, etc.</p>")

        with gr.Row():
            with gr.Column():
                ner_input = gr.Textbox(
                    label="Text 1",
                    info="Initial text",
                    lines=3,
                    value="A man named Arshad Chaudhary lives in Delhi. He works at Google.",
                )
                with gr.Row():
                    ner_clr = gr.Button("Clear")
                    ner_submit = gr.Button("Submit")
            ner_output = gr.HighlightedText(
                label="Diff",
                combine_adjacent=True,
                color_map={"PER": "red", "LOC": "green", "ORG": "blue", "MISC": "grey"})

        ner_submit.click(ner,inputs=ner_input,outputs=ner_output)
        gr.Examples(examples=["My name is Abdul Majid . I live in Lucknow and study at Integral University."],
                    inputs=ner_input)
    with gr.Tab("Zero Shot Qualification"):
        gr.HTML("<h1>Zero Shot Qualification</h1><br/>"
                "<p>Zero-shot learning is a type of learning that allows you to predict the class of a sample")
        with gr.Row():
            with gr.Column():
                zsq_input1 = gr.Textbox(
                    label="Text",
                    info="Initial text",
                    lines=3,
                )
                zsq_input2 = gr.Textbox(
                    label="Categories",
                    info="Categories to classify",
                    lines=3,
                    value="animal, human, vehicle",
                )
                with gr.Row():
                    zsq_clr = gr.Button("Clear")
                    zsq_submit = gr.Button("Submit")
            zsq_output = gr.JSON(label="Output")
            zsq_submit.click(zero_shot_classification,inputs=[zsq_input1, zsq_input2],outputs=zsq_output)
        gr.Examples(examples=demodata.zsq_examples,inputs=[zsq_input1, zsq_input2])
    with gr.Tab("Summarise"):
        gr.HTML("<h1>Summarise</h1><br/>"
                "<p>Summarization is the task of summarizing a document or a set of documents into a shorter version while retaining the most important information.</p>")
        with gr.Column():
            sum_input = gr.Textbox(
                label="Text",
                info="Text to summarise",
                lines=3,
            )
            with gr.Row():
                sum_clr = gr.Button("Clear")
                sum_submit = gr.Button("Submit")
            sum_output = gr.Textbox(label="Summary", lines=3)
        sum_submit.click(summarise,inputs=sum_input,outputs=sum_output)
        gr.Examples(examples=demodata.summarize_examples,inputs=sum_input)
    with gr.Tab("Document QnA"):
        gr.HTML("<h1>Document QnA</h1><br/>"
                "<p>Document Question Answering is a task of answering questions based on a given document. It gives short one word/phrase answers.</p>")
        with gr.Row():
            with gr.Column():
                doc_qna_input1 = gr.Image(
                    type="filepath",
                label="Image",
                height=512,
                width=512,
                )
                doc_qna_input2 = gr.Textbox(
                    label="Question",
                    info="Question to ask",
                    lines=1)
                with gr.Row():
                    doc_qna_clr = gr.Button("Clear")
                    doc_qna_submit = gr.Button("Submit")
            doc_qna_output = gr.Textbox(label="Answer",lines=2)
        doc_qna_submit.click(document_qna,inputs=[doc_qna_input1, doc_qna_input2],outputs=doc_qna_output)



demo.launch()
