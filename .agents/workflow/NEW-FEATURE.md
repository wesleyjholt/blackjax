## AI-assisted engineering workflow (concise)

If we are building a new feature/skill or developing code from scratch, follow these steps:

1. In a `reference` folder, create a new folder with the name of the feature. If building a skill, then the reference folder will be inside the skill's folder. All markdown files relevant to planning/code generation will live in this reference folder.
2. Write a SPEC.md file containing these explicit sections: scope, functional/non-functional requirements, constraints.
3. Ask the user 5-10 clarifying questions to make sure you understand what they want. Make sure to clarify what should and shouldn't be put inside a reusable software package.
4. Update the spec document.
5. (Repeat 2 and 3 until the specs are specific enough to ensure reproducible code generation.)
6. Write a CONTRACT.md file containing the following explicit sections: data models, API signatures, module boundaries.
7. Create a `steps` folder. Break the code generation process into 5 steps and write markdown file(s) with VERY detailed instructions for each step: 
    (a) Write tests. (Potentially several detailed .md files. Explicitely state the tests that should be written. Every function/class/module should have tests.)
    (b) Write source code. (Break this down into substeps. Make several detailed .md files for this step.)
    (c) Tie up loose ends. (One .md file. Be specific about what loose ends might need to be tied up.)
    (d) Add clean documentation (e.g., docstrings, type hints). (One .md file)
    (e) Add 1-3 easy-to-understand examples showcasing the new code/feature. These can be scripts or interactive notebooks. (Potentially several .md files. Be specific about the examples and what they should entail.)
8. Make sure the markdown files (especially the steps) are well-written, easy to follow, and provide enough detail so that multiple runs of code generation will produce practically the same or similar code.
9. Follow the implementation steps (in order) and spawn subagents as appropriate to handle concurrent tasks. Don't stop to ask questions unless you need clarification (e.g., as in step 3).