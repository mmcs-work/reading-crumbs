const notes = [
  {
    id: "creative-act-1",
    number: "1.1",
    title: "Make a place for the work",
    date: "2024-09-17",
    book: "The Creative Act",
    chapter: "Chapter 1",
    summary: "A creative practice becomes real when it has a small, repeatable place in your day. Protect the ritual before you optimize it.",
    points: [
      "A recurring time is more useful than waiting for motivation.",
      "Small constraints can make starting easier.",
      "The first goal is to return tomorrow, not to produce a masterpiece today.",
    ],
  },
  {
    id: "creative-act-2",
    number: "1.2",
    title: "Let the work surprise you",
    date: "2024-09-18",
    book: "The Creative Act",
    chapter: "Chapter 2",
    summary: "The best ideas often arrive after you have made room for them. The practice is less about forcing an outcome and more about staying available.",
    points: [
      "Make space before trying to make something good.",
      "The process is allowed to change the original idea.",
      "Attention is a creative tool, not just a productivity skill.",
    ],
  },
  {
    id: "four-thousand-weeks-1",
    number: "2.1",
    title: "Choose what to miss",
    date: "2024-09-12",
    book: "Four Thousand Weeks",
    chapter: "Chapter 5",
    summary: "A meaningful life is shaped as much by the things we decline as by the things we accomplish. Attention needs a boundary.",
    points: [
      "Every choice to focus is also a choice to leave something out.",
      "Trying to fit everything in is a way of avoiding commitment.",
    ],
  },
];

function Note({ note }: { note: (typeof notes)[number] }) {
  return (
    <section className="doc-section" id={note.id}>
      <h2><a href={`#${note.id}`}>{note.number}. {note.title}</a></h2>
      <p className="note-meta"><time dateTime={note.date}>{note.date}</time> · {note.book} · {note.chapter}</p>
      <p>{note.summary}</p>
      <h3>Interesting points</h3>
      <ul>{note.points.map((point) => <li key={point}>{point}</li>)}</ul>
      <blockquote>“The part worth remembering is the part that changes how I read the next page.”</blockquote>
    </section>
  );
}

export default function Home() {
  return (
    <main className="mdbook-shell">
      <aside className="sidebar" aria-label="Table of contents">
        <div className="sidebar-scrollbox">
          <a className="chapter-title" href="#top">My Reading Notes</a>
          <nav className="chapter">
          <a href="#introduction">Introduction</a>
          <a href="#how-this-works">How this works</a>
          <a className="chapter-item" href="#the-creative-act"><strong>1.</strong> The Creative Act</a>
          <a className="sub-chapter" href="#creative-act-1"><strong>1.1.</strong> Make a place for the work</a>
          <a className="sub-chapter" href="#creative-act-2"><strong>1.2.</strong> Let the work surprise you</a>
          <a className="chapter-item" href="#four-thousand-weeks"><strong>2.</strong> Four Thousand Weeks</a>
          <a className="sub-chapter" href="#four-thousand-weeks-1"><strong>2.1.</strong> Choose what to miss</a>
          <a className="chapter-item" href="#bookshelf">3. Bookshelf</a>
          <a className="chapter-item" href="#about">4. About</a>
          </nav>
        </div>
      </aside>

      <div className="page-wrapper" id="top">
        <header id="menu-bar">
          <button className="icon-button sidebar-toggle" type="button" aria-label="Toggle table of contents">☰</button>
          <span className="menu-title">My Reading Notes</span>
          <div className="menu-bar-actions"><button className="icon-button" type="button" aria-label="Change theme">◐</button><button className="icon-button" type="button" aria-label="Search notes">⌕</button></div>
        </header>

        <article id="content" className="content">
          <div className="page">
          <h1 className="chapter" id="introduction"><a href="#introduction">My Reading Notes</a></h1>
          <p className="lead">A running notebook of the books I am reading, the chapters I have finished, and the ideas that are worth carrying forward.</p>
          <p>This is a simple record rather than a review site. Each entry is added after a reading session and keeps three things together: where I stopped, what I found interesting, and what I want to remember.</p>

          <h2 id="how-this-works"><a className="header" href="#how-this-works">How this works</a></h2>
          <p>Notes are grouped by book and numbered in the order I read them. The navigation on the left is the index for the notebook.</p>
          <ol>
            <li>Read a chapter or a small section.</li>
            <li>Write down the idea that stayed with me.</li>
            <li>Add a few specific points, questions, or connections.</li>
          </ol>

          <blockquote><p><strong>Current habit:</strong> read for at least twenty minutes, then write before opening another tab.</p></blockquote>

          <h2 id="the-creative-act"><a className="header" href="#the-creative-act">1. The Creative Act</a></h2>
          <p className="book-description">Rick Rubin · currently reading</p>
          <Note note={notes[0]} />
          <Note note={notes[1]} />

          <h2 id="four-thousand-weeks"><a className="header" href="#four-thousand-weeks">2. Four Thousand Weeks</a></h2>
          <p className="book-description">Oliver Burkeman · currently reading</p>
          <Note note={notes[2]} />

          <h2 id="bookshelf"><a className="header" href="#bookshelf">Bookshelf</a></h2>
          <table><thead><tr><th>Book</th><th>Author</th><th>Status</th></tr></thead><tbody><tr><td>The Creative Act</td><td>Rick Rubin</td><td>Reading</td></tr><tr><td>Four Thousand Weeks</td><td>Oliver Burkeman</td><td>Reading</td></tr><tr><td>Braiding Sweetgrass</td><td>Robin Wall Kimmerer</td><td>Finished</td></tr></tbody></table>

          <h2 id="about"><a className="header" href="#about">About</a></h2>
          <p>This notebook is built to be updated slowly. The intended home for it is a small static site, with each new reading entry eventually living in a Markdown file that can be published through GitHub Pages.</p>
          <pre><code>_posts/2024-09-18-the-creative-act.md</code></pre>
          </div>
        </article>

        <footer className="page-nav"><a className="prev" href="#top">‹ My Reading Notes</a><a className="next" href="#bookshelf">Bookshelf ›</a></footer>
      </div>
    </main>
  );
}
